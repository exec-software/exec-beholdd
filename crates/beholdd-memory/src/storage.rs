//! Conservative checkpoint-storage classification for Linux.
//!
//! The probe only reads the calling process's mount table and the kernel's block-device
//! rotational flag.  A result is considered suitable for streaming by default only when
//! every shard can be traced to a local block device that reports itself non-rotational.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::os::unix::ffi::OsStringExt;
use std::path::{Path, PathBuf};

/// Coarse storage class relevant to checkpoint streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    /// Every shard is on a local block device reporting `rotational = 0`.
    LocalSolidState,
    /// Every shard is on a local block device reporting a non-zero rotational flag.
    Rotational,
    /// Every shard is on a known network filesystem.
    Network,
    /// Shards resolve to more than one storage class.
    Mixed,
    /// The available mount or block-device evidence is insufficient.
    Unknown,
}

impl StorageClass {
    /// Whether this class is accepted without an explicit override.
    pub const fn is_safe_for_streaming(self) -> bool {
        matches!(self, Self::LocalSolidState)
    }
}

impl fmt::Display for StorageClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::LocalSolidState => "local solid-state",
            Self::Rotational => "rotational",
            Self::Network => "network-backed",
            Self::Mixed => "mixed",
            Self::Unknown => "unknown",
        })
    }
}

/// Explicit choice controlling whether unsuitable checkpoint storage may be used.
///
/// The default is deliberately conservative.  Constructing [`StorageOverride::AllowUnsafe`]
/// is the opt-in required to run from rotational, network-backed, mixed, or unidentifiable
/// storage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum StorageOverride {
    /// Require every checkpoint shard to be on identifiable local solid-state storage.
    #[default]
    RefuseUnsafe,
    /// Continue despite storage that is slow or whose characteristics cannot be proven.
    AllowUnsafe,
}

/// Storage evidence collected for one checkpoint shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardStorage {
    path: PathBuf,
    resolved_path: Option<PathBuf>,
    class: StorageClass,
    mount_point: Option<PathBuf>,
    filesystem: Option<String>,
    device: Option<String>,
    detail: String,
}

impl ShardStorage {
    /// Path supplied by the caller.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Canonical path used for mount matching, if the path could be resolved.
    pub fn resolved_path(&self) -> Option<&Path> {
        self.resolved_path.as_deref()
    }

    /// Classification of this shard.
    pub const fn class(&self) -> StorageClass {
        self.class
    }

    /// Most-specific mount point containing the resolved shard.
    pub fn mount_point(&self) -> Option<&Path> {
        self.mount_point.as_deref()
    }

    /// Filesystem type reported by `/proc/self/mountinfo`.
    pub fn filesystem(&self) -> Option<&str> {
        self.filesystem.as_deref()
    }

    /// Kernel block-device number, formatted as `major:minor`, when applicable.
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    /// Human-readable evidence or reason why the shard could not be classified.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Aggregate storage classification and the evidence for every shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageAssessment {
    class: StorageClass,
    shards: Vec<ShardStorage>,
}

impl StorageAssessment {
    /// Aggregate class.  Differently classified shards produce [`StorageClass::Mixed`].
    pub const fn class(&self) -> StorageClass {
        self.class
    }

    /// Per-shard evidence, in the same order as the input paths.
    pub fn shards(&self) -> &[ShardStorage] {
        &self.shards
    }
}

/// Reason a checkpoint-storage safety check was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageRefusal {
    /// No checkpoint shard paths were supplied, so there was nothing to classify.
    NoShards,
    /// Storage did not meet the default local-solid-state requirement.
    Unsuitable(StorageClass),
}

/// Typed refusal returned before streaming from unsuitable checkpoint storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageSafetyError {
    refusal: StorageRefusal,
    assessment: StorageAssessment,
}

impl StorageSafetyError {
    /// Machine-readable reason for the refusal.
    pub const fn refusal(&self) -> StorageRefusal {
        self.refusal
    }

    /// Complete assessment that caused the refusal.
    pub const fn assessment(&self) -> &StorageAssessment {
        &self.assessment
    }
}

impl fmt::Display for StorageSafetyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.refusal {
            StorageRefusal::NoShards => formatter.write_str(
                "checkpoint storage cannot be checked because no shard paths were supplied",
            ),
            StorageRefusal::Unsuitable(class) => {
                write!(
                    formatter,
                    "checkpoint storage is {class}; streaming requires local solid-state storage by default"
                )?;

                if let Some(shard) = self
                    .assessment
                    .shards
                    .iter()
                    .find(|shard| !shard.class.is_safe_for_streaming())
                {
                    write!(formatter, ": {} ({})", shard.path.display(), shard.detail)?;
                }

                formatter.write_str(
                    "; move every shard to local solid-state storage, or explicitly opt in with StorageOverride::AllowUnsafe",
                )
            }
        }
    }
}

impl std::error::Error for StorageSafetyError {}

/// Classify all checkpoint shards using `/proc/self/mountinfo` and Linux sysfs.
///
/// Probe failures are represented as [`StorageClass::Unknown`] evidence rather than being
/// treated as permission to stream.  Paths are canonicalized before matching so a symlink
/// into a different mount cannot inherit the classification of its parent directory.
pub fn classify_checkpoint_storage<I, P>(shard_paths: I) -> StorageAssessment
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    classify_with_probe(shard_paths, &LinuxStorageProbe)
}

/// Apply the default checkpoint-storage safety policy.
///
/// Only an assessment of [`StorageClass::LocalSolidState`] succeeds.  Use
/// [`check_checkpoint_storage`] when the caller exposes an explicit unsafe-storage opt-in.
pub fn require_safe_checkpoint_storage<I, P>(
    shard_paths: I,
) -> Result<StorageAssessment, StorageSafetyError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    check_checkpoint_storage(shard_paths, StorageOverride::default())
}

/// Classify checkpoint shards and enforce the requested storage override.
///
/// An empty shard list is always rejected.  [`StorageOverride::AllowUnsafe`] permits all
/// non-empty classifications, including unknown storage, because its purpose is to make
/// that risk acceptance explicit at the call site.
pub fn check_checkpoint_storage<I, P>(
    shard_paths: I,
    storage_override: StorageOverride,
) -> Result<StorageAssessment, StorageSafetyError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let assessment = classify_checkpoint_storage(shard_paths);
    enforce_assessment(assessment, storage_override)
}

fn enforce_assessment(
    assessment: StorageAssessment,
    storage_override: StorageOverride,
) -> Result<StorageAssessment, StorageSafetyError> {
    let refusal = if assessment.shards.is_empty() {
        Some(StorageRefusal::NoShards)
    } else if assessment.class.is_safe_for_streaming()
        || storage_override == StorageOverride::AllowUnsafe
    {
        None
    } else {
        Some(StorageRefusal::Unsuitable(assessment.class))
    };

    match refusal {
        Some(refusal) => Err(StorageSafetyError {
            refusal,
            assessment,
        }),
        None => Ok(assessment),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BlockDevice {
    major: u32,
    minor: u32,
}

impl fmt::Display for BlockDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.major, self.minor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MountInfo {
    mount_id: u64,
    device: BlockDevice,
    mount_point: PathBuf,
    filesystem: String,
}

trait StorageProbe {
    fn mountinfo(&self) -> io::Result<String>;
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
    fn rotational(&self, device: BlockDevice) -> io::Result<String>;
}

struct LinuxStorageProbe;

impl StorageProbe for LinuxStorageProbe {
    fn mountinfo(&self) -> io::Result<String> {
        fs::read_to_string("/proc/self/mountinfo")
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        fs::canonicalize(path)
    }

    fn rotational(&self, device: BlockDevice) -> io::Result<String> {
        let device_path = PathBuf::from(format!("/sys/dev/block/{device}"));
        let resolved = fs::canonicalize(device_path)?;

        // A partition normally has no queue directory of its own.  Its parent block
        // device does, while stacked devices generally expose a queue directly.
        for ancestor in resolved
            .ancestors()
            .take_while(|path| path.starts_with("/sys"))
        {
            let rotational = ancestor.join("queue/rotational");
            match fs::read_to_string(&rotational) {
                Ok(value) => return Ok(value),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no queue/rotational attribute for block device {device}"),
        ))
    }
}

fn classify_with_probe<I, P, T>(shard_paths: I, probe: &T) -> StorageAssessment
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
    T: StorageProbe,
{
    let paths: Vec<PathBuf> = shard_paths
        .into_iter()
        .map(|path| path.as_ref().to_owned())
        .collect();

    if paths.is_empty() {
        return StorageAssessment {
            class: StorageClass::Unknown,
            shards: Vec::new(),
        };
    }

    let mounts = match probe.mountinfo() {
        Ok(contents) => match parse_mountinfo(&contents) {
            Ok(mounts) => mounts,
            Err(reason) => {
                return unknown_assessment(paths, format!("could not parse mount table: {reason}"));
            }
        },
        Err(error) => {
            return unknown_assessment(paths, format!("could not read mount table: {error}"));
        }
    };

    let shards = paths
        .into_iter()
        .map(|path| classify_shard(path, &mounts, probe))
        .collect();
    aggregate(shards)
}

fn unknown_assessment(paths: Vec<PathBuf>, detail: String) -> StorageAssessment {
    StorageAssessment {
        class: StorageClass::Unknown,
        shards: paths
            .into_iter()
            .map(|path| ShardStorage {
                path,
                resolved_path: None,
                class: StorageClass::Unknown,
                mount_point: None,
                filesystem: None,
                device: None,
                detail: detail.clone(),
            })
            .collect(),
    }
}

fn classify_shard<T: StorageProbe>(path: PathBuf, mounts: &[MountInfo], probe: &T) -> ShardStorage {
    let resolved_path = match probe.canonicalize(&path) {
        Ok(path) => path,
        Err(error) => {
            return ShardStorage {
                path,
                resolved_path: None,
                class: StorageClass::Unknown,
                mount_point: None,
                filesystem: None,
                device: None,
                detail: format!("could not resolve shard path: {error}"),
            };
        }
    };

    let Some(mount) = mounts
        .iter()
        .filter(|mount| resolved_path.starts_with(&mount.mount_point))
        .max_by_key(|mount| {
            (
                mount.mount_point.components().count(),
                mount.mount_point.as_os_str().len(),
                mount.mount_id,
            )
        })
    else {
        return ShardStorage {
            path,
            resolved_path: Some(resolved_path),
            class: StorageClass::Unknown,
            mount_point: None,
            filesystem: None,
            device: None,
            detail: "no containing mount was found".to_owned(),
        };
    };

    let filesystem = Some(mount.filesystem.clone());
    let mount_point = Some(mount.mount_point.clone());

    if is_network_filesystem(&mount.filesystem) {
        return ShardStorage {
            path,
            resolved_path: Some(resolved_path),
            class: StorageClass::Network,
            mount_point,
            filesystem,
            device: None,
            detail: format!(
                "filesystem {} at {} is network-backed",
                mount.filesystem,
                mount.mount_point.display()
            ),
        };
    }

    let device = mount.device.to_string();
    match probe
        .rotational(mount.device)
        .and_then(|contents| parse_rotational(&contents))
    {
        Ok(false) => ShardStorage {
            path,
            resolved_path: Some(resolved_path),
            class: StorageClass::LocalSolidState,
            mount_point,
            filesystem,
            device: Some(device),
            detail: format!("local block device {} reports rotational=0", mount.device),
        },
        Ok(true) => ShardStorage {
            path,
            resolved_path: Some(resolved_path),
            class: StorageClass::Rotational,
            mount_point,
            filesystem,
            device: Some(device),
            detail: format!(
                "local block device {} reports a non-zero rotational flag",
                mount.device
            ),
        },
        Err(error) => ShardStorage {
            path,
            resolved_path: Some(resolved_path),
            class: StorageClass::Unknown,
            mount_point,
            filesystem,
            device: Some(device),
            detail: format!(
                "could not determine whether block device {} is rotational: {error}",
                mount.device
            ),
        },
    }
}

fn aggregate(shards: Vec<ShardStorage>) -> StorageAssessment {
    let class = shards
        .first()
        .map(|first| {
            if shards.iter().all(|shard| shard.class == first.class) {
                first.class
            } else {
                StorageClass::Mixed
            }
        })
        .unwrap_or(StorageClass::Unknown);

    StorageAssessment { class, shards }
}

fn parse_rotational(contents: &str) -> io::Result<bool> {
    let value = contents.trim().parse::<u32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid queue/rotational value {contents:?}"),
        )
    })?;
    Ok(value != 0)
}

fn is_network_filesystem(filesystem: &str) -> bool {
    matches!(
        filesystem.to_ascii_lowercase().as_str(),
        "9p" | "afs"
            | "ceph"
            | "cifs"
            | "coda"
            | "davfs"
            | "fuse.ceph"
            | "fuse.davfs"
            | "fuse.glusterfs"
            | "fuse.gcsfuse"
            | "fuse.rclone"
            | "fuse.s3fs"
            | "fuse.sshfs"
            | "glusterfs"
            | "gfs"
            | "gfs2"
            | "lustre"
            | "ncp"
            | "ncpfs"
            | "nfs"
            | "nfs4"
            | "ocfs2"
            | "orangefs"
            | "pvfs2"
            | "smb3"
    )
}

fn parse_mountinfo(contents: &str) -> Result<Vec<MountInfo>, MountInfoParseError> {
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| parse_mountinfo_line(index + 1, line))
        .collect()
}

fn parse_mountinfo_line(line_number: usize, line: &str) -> Result<MountInfo, MountInfoParseError> {
    let invalid = |reason: &str| MountInfoParseError {
        line: line_number,
        reason: reason.to_owned(),
    };

    let (mount_fields, filesystem_fields) = line
        .split_once(" - ")
        .ok_or_else(|| invalid("missing ` - ` field separator"))?;
    let mut mount_fields = mount_fields.split_whitespace();
    let mount_id = mount_fields
        .next()
        .ok_or_else(|| invalid("missing mount ID"))?
        .parse::<u64>()
        .map_err(|_| invalid("invalid mount ID"))?;
    mount_fields
        .next()
        .ok_or_else(|| invalid("missing parent mount ID"))?;
    let device = parse_device(
        mount_fields
            .next()
            .ok_or_else(|| invalid("missing block-device number"))?,
    )
    .map_err(&invalid)?;
    mount_fields
        .next()
        .ok_or_else(|| invalid("missing mount root"))?;
    let mount_point = decode_mount_path(
        mount_fields
            .next()
            .ok_or_else(|| invalid("missing mount point"))?,
    )
    .map_err(&invalid)?;

    let filesystem = filesystem_fields
        .split_whitespace()
        .next()
        .ok_or_else(|| invalid("missing filesystem type"))?
        .to_owned();

    Ok(MountInfo {
        mount_id,
        device,
        mount_point,
        filesystem,
    })
}

fn parse_device(value: &str) -> Result<BlockDevice, &'static str> {
    let (major, minor) = value.split_once(':').ok_or("invalid block-device number")?;
    let major = major
        .parse::<u32>()
        .map_err(|_| "invalid block-device major number")?;
    let minor = minor
        .parse::<u32>()
        .map_err(|_| "invalid block-device minor number")?;
    Ok(BlockDevice { major, minor })
}

pub(crate) fn decode_mount_path(value: &str) -> Result<PathBuf, &'static str> {
    let input = value.as_bytes();
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        if input[index] != b'\\' {
            output.push(input[index]);
            index += 1;
            continue;
        }

        let Some(octal) = input.get(index + 1..index + 4) else {
            return Err("truncated mount-point escape");
        };
        if !octal.iter().all(|byte| matches!(byte, b'0'..=b'7')) {
            return Err("invalid mount-point escape");
        }
        let decoded = u16::from(octal[0] - b'0') * 64
            + u16::from(octal[1] - b'0') * 8
            + u16::from(octal[2] - b'0');
        let decoded = u8::try_from(decoded).map_err(|_| "mount-point escape exceeds one byte")?;
        output.push(decoded);
        index += 4;
    }

    Ok(PathBuf::from(OsString::from_vec(output)))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MountInfoParseError {
    line: usize,
    reason: String,
}

impl fmt::Display for MountInfoParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {}: {}", self.line, self.reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    struct SyntheticProbe {
        mountinfo: io::Result<String>,
        unresolved: HashSet<PathBuf>,
        rotational: HashMap<BlockDevice, io::Result<String>>,
    }

    impl SyntheticProbe {
        fn new(mountinfo: &str) -> Self {
            Self {
                mountinfo: Ok(mountinfo.to_owned()),
                unresolved: HashSet::new(),
                rotational: HashMap::new(),
            }
        }

        fn with_rotational(mut self, device: BlockDevice, value: &str) -> Self {
            self.rotational.insert(device, Ok(value.to_owned()));
            self
        }
    }

    impl StorageProbe for SyntheticProbe {
        fn mountinfo(&self) -> io::Result<String> {
            match &self.mountinfo {
                Ok(value) => Ok(value.clone()),
                Err(error) => Err(io::Error::new(error.kind(), error.to_string())),
            }
        }

        fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
            if self.unresolved.contains(path) {
                Err(io::Error::new(io::ErrorKind::NotFound, "synthetic miss"))
            } else {
                Ok(path.to_owned())
            }
        }

        fn rotational(&self, device: BlockDevice) -> io::Result<String> {
            match self.rotational.get(&device) {
                Some(Ok(value)) => Ok(value.clone()),
                Some(Err(error)) => Err(io::Error::new(error.kind(), error.to_string())),
                None => Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "synthetic sysfs miss",
                )),
            }
        }
    }

    fn device(major: u32, minor: u32) -> BlockDevice {
        BlockDevice { major, minor }
    }

    #[test]
    fn parses_mountinfo_optional_fields_and_escaped_mount_point() {
        let mounts = parse_mountinfo(
            "41 30 259:2 / /mnt/my\\040models rw,relatime shared:7 master:1 - ext4 /dev/nvme0n1p2 rw\n",
        )
        .unwrap();

        assert_eq!(
            mounts,
            vec![MountInfo {
                mount_id: 41,
                device: device(259, 2),
                mount_point: PathBuf::from("/mnt/my models"),
                filesystem: "ext4".to_owned(),
            }]
        );
    }

    #[test]
    fn malformed_mountinfo_is_reported_instead_of_partially_trusted() {
        let error = parse_mountinfo("41 30 nope / /mnt rw - ext4 /dev/x rw\n").unwrap_err();
        assert_eq!(error.line, 1);
        assert!(error.reason.contains("block-device"));

        let error = parse_mountinfo("41 30 8:1 / /bad\\09x rw - ext4 /dev/x rw\n").unwrap_err();
        assert!(error.reason.contains("escape"));

        let error = parse_mountinfo("41 30 8:1 / /bad\\777 rw - ext4 /dev/x rw\n").unwrap_err();
        assert!(error.reason.contains("exceeds one byte"));
    }

    #[test]
    fn parses_synthetic_sysfs_rotational_values_conservatively() {
        assert!(!parse_rotational("0\n").unwrap());
        assert!(parse_rotational("1\n").unwrap());
        assert!(parse_rotational("2").unwrap());
        assert!(parse_rotational("maybe\n").is_err());
    }

    #[test]
    fn classifies_local_solid_state_from_the_most_specific_mount() {
        let probe = SyntheticProbe::new(
            "20 1 8:1 / / rw - ext4 /dev/sda1 rw\n\
             21 20 259:2 / /models rw - xfs /dev/nvme0n1p2 rw\n",
        )
        .with_rotational(device(8, 1), "1\n")
        .with_rotational(device(259, 2), "0\n");

        let assessment = classify_with_probe(["/models/weights/a.gguf"], &probe);
        assert_eq!(assessment.class(), StorageClass::LocalSolidState);
        assert_eq!(
            assessment.shards()[0].mount_point(),
            Some(Path::new("/models"))
        );
        assert_eq!(assessment.shards()[0].filesystem(), Some("xfs"));
        assert_eq!(assessment.shards()[0].device(), Some("259:2"));
    }

    #[test]
    fn mount_matching_observes_path_component_boundaries() {
        let probe = SyntheticProbe::new(
            "20 1 8:1 / / rw - ext4 /dev/sda1 rw\n\
             21 20 259:2 / /models rw - xfs /dev/nvme0n1p2 rw\n",
        )
        .with_rotational(device(8, 1), "1")
        .with_rotational(device(259, 2), "0");

        let assessment = classify_with_probe(["/models-old/a.gguf"], &probe);
        assert_eq!(assessment.class(), StorageClass::Rotational);
        assert_eq!(assessment.shards()[0].mount_point(), Some(Path::new("/")));
    }

    #[test]
    fn known_network_filesystems_do_not_trust_a_block_device_flag() {
        let probe = SyntheticProbe::new("30 1 0:52 / /models rw - nfs4 server:/models rw\n")
            .with_rotational(device(0, 52), "0");

        let assessment = classify_with_probe(["/models/a.gguf"], &probe);
        assert_eq!(assessment.class(), StorageClass::Network);
        assert_eq!(assessment.shards()[0].filesystem(), Some("nfs4"));
        assert_eq!(assessment.shards()[0].device(), None);
    }

    #[test]
    fn differently_backed_shards_are_mixed_and_each_keeps_evidence() {
        let probe = SyntheticProbe::new(
            "20 1 259:2 / /models rw - ext4 /dev/nvme0n1p2 rw\n\
             30 1 0:52 / /remote rw - cifs //server/models rw\n",
        )
        .with_rotational(device(259, 2), "0");

        let assessment = classify_with_probe(
            ["/models/model-00001.gguf", "/remote/model-00002.gguf"],
            &probe,
        );
        assert_eq!(assessment.class(), StorageClass::Mixed);
        assert_eq!(assessment.shards().len(), 2);
        assert_eq!(
            assessment.shards()[0].class(),
            StorageClass::LocalSolidState
        );
        assert_eq!(assessment.shards()[1].class(), StorageClass::Network);
    }

    #[test]
    fn missing_or_invalid_sysfs_evidence_is_unknown() {
        let mountinfo = "20 1 7:4 / /models rw - ext4 /dev/loop4 rw\n";
        let missing = classify_with_probe(["/models/a.gguf"], &SyntheticProbe::new(mountinfo));
        assert_eq!(missing.class(), StorageClass::Unknown);
        assert!(missing.shards()[0]
            .detail()
            .contains("synthetic sysfs miss"));

        let invalid = classify_with_probe(
            ["/models/a.gguf"],
            &SyntheticProbe::new(mountinfo).with_rotational(device(7, 4), "unknown"),
        );
        assert_eq!(invalid.class(), StorageClass::Unknown);
        assert!(invalid.shards()[0]
            .detail()
            .contains("invalid queue/rotational"));
    }

    #[test]
    fn unresolvable_paths_and_unreadable_mountinfo_are_unknown() {
        let mut unresolved =
            SyntheticProbe::new("20 1 259:2 / /models rw - ext4 /dev/nvme0n1p2 rw\n");
        unresolved
            .unresolved
            .insert(PathBuf::from("/models/missing.gguf"));
        let assessment = classify_with_probe(["/models/missing.gguf"], &unresolved);
        assert_eq!(assessment.class(), StorageClass::Unknown);
        assert!(assessment.shards()[0].resolved_path().is_none());

        let unreadable = SyntheticProbe {
            mountinfo: Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied")),
            unresolved: HashSet::new(),
            rotational: HashMap::new(),
        };
        let assessment = classify_with_probe(["/models/a.gguf"], &unreadable);
        assert_eq!(assessment.class(), StorageClass::Unknown);
        assert!(assessment.shards()[0].detail().contains("denied"));
    }

    #[test]
    fn default_policy_refuses_every_non_solid_state_class() {
        for class in [
            StorageClass::Rotational,
            StorageClass::Network,
            StorageClass::Mixed,
            StorageClass::Unknown,
        ] {
            let assessment = StorageAssessment {
                class,
                shards: vec![ShardStorage {
                    path: PathBuf::from("/models/a.gguf"),
                    resolved_path: Some(PathBuf::from("/models/a.gguf")),
                    class,
                    mount_point: None,
                    filesystem: None,
                    device: None,
                    detail: "synthetic evidence".to_owned(),
                }],
            };
            let error = enforce_assessment(assessment, StorageOverride::default()).unwrap_err();
            assert_eq!(error.refusal(), StorageRefusal::Unsuitable(class));
            assert!(error.to_string().contains("move every shard"));
            assert!(error.to_string().contains("AllowUnsafe"));
        }
    }

    #[test]
    fn explicit_override_allows_unsafe_storage_but_not_an_empty_input() {
        let network = StorageAssessment {
            class: StorageClass::Network,
            shards: vec![ShardStorage {
                path: PathBuf::from("/models/a.gguf"),
                resolved_path: Some(PathBuf::from("/models/a.gguf")),
                class: StorageClass::Network,
                mount_point: Some(PathBuf::from("/models")),
                filesystem: Some("nfs4".to_owned()),
                device: None,
                detail: "synthetic network".to_owned(),
            }],
        };
        assert!(enforce_assessment(network, StorageOverride::AllowUnsafe).is_ok());

        let empty = StorageAssessment {
            class: StorageClass::Unknown,
            shards: Vec::new(),
        };
        let error = enforce_assessment(empty, StorageOverride::AllowUnsafe).unwrap_err();
        assert_eq!(error.refusal(), StorageRefusal::NoShards);
    }
}
