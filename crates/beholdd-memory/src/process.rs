//! Measured Linux process and cgroup residency plus available-memory discovery.
//!
//! Fit plans are forecasts. This module reads the kernel's observed current and peak
//! resident sets so callers can report the measurements beside the forecast. It also
//! combines Linux `MemAvailable` with every finite cgroup-v2 ancestor limit so a fit
//! decision cannot accidentally budget memory the process is not allowed to use.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::storage::decode_mount_path;

/// Current and peak resident bytes reported by Linux for this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    resident_bytes: u64,
    peak_resident_bytes: u64,
}

/// Current and peak bytes charged to the calling process's cgroup-v2 leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CgroupMemory {
    current_bytes: u64,
    peak_bytes: u64,
}

/// Host and cgroup memory headroom visible to the calling process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvailableMemory {
    host_available_bytes: u64,
    cgroup_available_bytes: Option<u64>,
}

impl AvailableMemory {
    /// Linux `MemAvailable`, before applying a process cgroup ceiling.
    pub const fn host_available_bytes(self) -> u64 {
        self.host_available_bytes
    }

    /// Tightest finite cgroup-v2 ancestor headroom, when one exists.
    pub const fn cgroup_available_bytes(self) -> Option<u64> {
        self.cgroup_available_bytes
    }

    /// The smaller of host availability and finite cgroup headroom.
    pub const fn effective_available_bytes(self) -> u64 {
        match self.cgroup_available_bytes {
            Some(bytes) if bytes < self.host_available_bytes => bytes,
            Some(_) | None => self.host_available_bytes,
        }
    }
}

impl ProcessMemory {
    /// Bytes resident when `/proc/self/status` was sampled.
    pub const fn resident_bytes(self) -> u64 {
        self.resident_bytes
    }

    /// Maximum resident bytes observed by the kernel over this process's lifetime.
    pub const fn peak_resident_bytes(self) -> u64 {
        self.peak_resident_bytes
    }
}

impl CgroupMemory {
    /// Bytes charged to the unified cgroup when `memory.current` was sampled.
    pub const fn current_bytes(self) -> u64 {
        self.current_bytes
    }

    /// Maximum bytes charged since the cgroup was created or `memory.peak` was reset.
    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

/// Failure to obtain an unambiguous process-residency measurement.
#[derive(Debug)]
pub enum ProcessMemoryError {
    /// `/proc/self/status` could not be read.
    Read(std::io::Error),
    /// A required status field was absent or malformed.
    InvalidStatus(String),
}

impl fmt::Display for ProcessMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(formatter, "reading /proc/self/status: {error}"),
            Self::InvalidStatus(reason) => {
                write!(
                    formatter,
                    "parsing process residency from /proc/self/status: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for ProcessMemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::InvalidStatus(_) => None,
        }
    }
}

/// Failure to obtain an unambiguous cgroup-v2 memory measurement.
#[derive(Debug)]
pub enum CgroupMemoryError {
    /// A Linux control file could not be read.
    Read {
        /// File whose read failed.
        path: PathBuf,
        /// Kernel or filesystem error.
        source: io::Error,
    },
    /// Linux exposed malformed, unsafe or internally inconsistent cgroup data.
    Invalid(String),
}

impl fmt::Display for CgroupMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "reading {}: {source}", path.display())
            }
            Self::Invalid(reason) => write!(formatter, "sampling cgroup memory: {reason}"),
        }
    }
}

impl std::error::Error for CgroupMemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

/// Failure to obtain an unambiguous host-and-cgroup memory allowance.
#[derive(Debug)]
pub enum AvailableMemoryError {
    /// A Linux control file could not be read.
    Read {
        /// File whose read failed.
        path: PathBuf,
        /// Kernel or filesystem error.
        source: io::Error,
    },
    /// Linux exposed malformed or internally inconsistent memory-control data.
    Invalid(String),
}

impl fmt::Display for AvailableMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "reading {}: {source}", path.display())
            }
            Self::Invalid(reason) => write!(formatter, "discovering available memory: {reason}"),
        }
    }
}

impl std::error::Error for AvailableMemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

/// Sample the calling process's current and peak resident sets.
pub fn process_memory() -> Result<ProcessMemory, ProcessMemoryError> {
    let status = fs::read_to_string("/proc/self/status").map_err(ProcessMemoryError::Read)?;
    parse_status(&status)
}

/// Sample the calling process's unified-cgroup leaf memory counters.
///
/// The leaf is resolved from `/proc/self/cgroup` through the matching cgroup2 mount root
/// in `/proc/self/mountinfo`; a mount-root path is never guessed. Systems with no unified
/// membership return `Ok(None)`. An active cgroup-v1 memory controller, including a
/// hybrid v1/v2 memory setup, is rejected because these counters would be incomplete.
pub fn cgroup_memory() -> Result<Option<CgroupMemory>, CgroupMemoryError> {
    let membership_path = Path::new("/proc/self/cgroup");
    let membership = read_cgroup_text(membership_path)?;
    let mountinfo_path = Path::new("/proc/self/mountinfo");
    let mountinfo = read_cgroup_text(mountinfo_path)?;

    cgroup_memory_from(&membership, &mountinfo)
}

/// Measure memory currently available to this process.
///
/// The cgroup-v2 lookup resolves the calling process's actual unified-cgroup path from
/// `/proc/self/cgroup` through the cgroup2 mount root in `/proc/self/mountinfo`. Every
/// ancestor between that cgroup and the mounted hierarchy root is checked because a
/// parent may impose the effective hard limit. Systems without a unified cgroup-v2
/// membership fall back to host `MemAvailable`.
pub fn available_memory() -> Result<AvailableMemory, AvailableMemoryError> {
    let meminfo_path = Path::new("/proc/meminfo");
    let meminfo = read_text(meminfo_path)?;
    let host_available_bytes =
        parse_available_memory(&meminfo).map_err(AvailableMemoryError::Invalid)?;

    let cgroup_path = Path::new("/proc/self/cgroup");
    let membership = read_text(cgroup_path)?;
    let mountinfo_path = Path::new("/proc/self/mountinfo");
    let mountinfo = read_text(mountinfo_path)?;
    let cgroup_available_bytes = match resolve_cgroup2_directory(&membership, &mountinfo)
        .map_err(AvailableMemoryError::Invalid)?
    {
        Some((directory, mount_point)) => cgroup_ancestor_headroom(&directory, &mount_point)?,
        None => None,
    };

    Ok(AvailableMemory {
        host_available_bytes,
        cgroup_available_bytes,
    })
}

fn read_text(path: &Path) -> Result<String, AvailableMemoryError> {
    fs::read_to_string(path).map_err(|source| AvailableMemoryError::Read {
        path: path.to_owned(),
        source,
    })
}

fn read_cgroup_text(path: &Path) -> Result<String, CgroupMemoryError> {
    fs::read_to_string(path).map_err(|source| CgroupMemoryError::Read {
        path: path.to_owned(),
        source,
    })
}

fn cgroup_memory_from(
    membership: &str,
    mountinfo: &str,
) -> Result<Option<CgroupMemory>, CgroupMemoryError> {
    let directory = match resolve_cgroup2_directory(membership, mountinfo)
        .map_err(CgroupMemoryError::Invalid)?
    {
        Some((directory, _mount_point)) => directory,
        None => return Ok(None),
    };
    sample_cgroup_memory(&directory).map(Some)
}

fn sample_cgroup_memory(directory: &Path) -> Result<CgroupMemory, CgroupMemoryError> {
    let current_path = directory.join("memory.current");
    let current_bytes = parse_cgroup_byte_count(&read_cgroup_text(&current_path)?, &current_path)?;
    let peak_path = directory.join("memory.peak");
    let peak_bytes = parse_cgroup_byte_count(&read_cgroup_text(&peak_path)?, &peak_path)?;
    if peak_bytes < current_bytes {
        return Err(CgroupMemoryError::Invalid(format!(
            "{} ({peak_bytes} bytes) is smaller than {} ({current_bytes} bytes)",
            peak_path.display(),
            current_path.display()
        )));
    }
    Ok(CgroupMemory {
        current_bytes,
        peak_bytes,
    })
}

fn parse_cgroup_byte_count(value: &str, path: &Path) -> Result<u64, CgroupMemoryError> {
    let value = value.strip_suffix('\n').unwrap_or(value);
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CgroupMemoryError::Invalid(format!(
            "{} must contain exactly one unsigned decimal byte count",
            path.display()
        )));
    }
    value.parse::<u64>().map_err(|_| {
        CgroupMemoryError::Invalid(format!("{} unsigned byte count overflows", path.display()))
    })
}

fn parse_available_memory(meminfo: &str) -> Result<u64, String> {
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .ok_or_else(|| "/proc/meminfo has no MemAvailable field".to_owned())?;
    let mut values = line["MemAvailable:".len()..].split_whitespace();
    let kib = values
        .next()
        .ok_or_else(|| "MemAvailable has no value".to_owned())?
        .parse::<u64>()
        .map_err(|_| "MemAvailable is not an unsigned integer".to_owned())?;
    if values.next() != Some("kB") || values.next().is_some() {
        return Err("MemAvailable must contain exactly one kB value".to_owned());
    }
    kib.checked_mul(1024)
        .ok_or_else(|| "MemAvailable byte count overflows".to_owned())
}

fn resolve_cgroup2_directory(
    membership: &str,
    mountinfo: &str,
) -> Result<Option<(PathBuf, PathBuf)>, String> {
    let mut unified_path = None;
    let mut legacy_memory_controller = false;
    for (index, line) in membership.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split(':');
        let hierarchy = fields
            .next()
            .ok_or_else(|| format!("/proc/self/cgroup line {} has no hierarchy", index + 1))?;
        let controllers = fields
            .next()
            .ok_or_else(|| format!("/proc/self/cgroup line {} has no controllers", index + 1))?;
        let path = fields
            .next()
            .ok_or_else(|| format!("/proc/self/cgroup line {} has no path", index + 1))?;
        if fields.next().is_some() || path.is_empty() {
            return Err(format!(
                "/proc/self/cgroup line {} is not hierarchy:controllers:path",
                index + 1
            ));
        }
        if controllers.is_empty() {
            if hierarchy != "0" {
                return Err(format!(
                    "/proc/self/cgroup line {} has an empty controller list outside hierarchy 0",
                    index + 1
                ));
            }
            if unified_path.replace(path).is_some() {
                return Err("/proc/self/cgroup contains multiple unified memberships".to_owned());
            }
        } else if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            legacy_memory_controller = true;
        }
    }
    if legacy_memory_controller {
        return Err(
            "a cgroup-v1 memory controller is active; its finite limit is not implemented"
                .to_owned(),
        );
    }
    let Some(unified_path) = unified_path else {
        return Ok(None);
    };
    let unified_path = PathBuf::from(unified_path);
    if !is_normal_absolute_path(&unified_path) {
        return Err(format!(
            "unified cgroup path {} is not a normal absolute path",
            unified_path.display()
        ));
    }

    let mut candidates = Vec::new();
    for (index, line) in mountinfo.lines().enumerate() {
        let Some((mount_fields, filesystem_fields)) = line.split_once(" - ") else {
            continue;
        };
        if filesystem_fields.split_whitespace().next() != Some("cgroup2") {
            continue;
        }
        let fields: Vec<&str> = mount_fields.split_whitespace().collect();
        if fields.len() < 5 {
            return Err(format!(
                "/proc/self/mountinfo line {} has incomplete cgroup2 fields",
                index + 1
            ));
        }
        let root = decode_mount_path(fields[3]).map_err(|reason| {
            format!(
                "/proc/self/mountinfo line {} has invalid cgroup2 root: {reason}",
                index + 1
            )
        })?;
        let mount_point = decode_mount_path(fields[4]).map_err(|reason| {
            format!(
                "/proc/self/mountinfo line {} has invalid cgroup2 mount point: {reason}",
                index + 1
            )
        })?;
        if !is_normal_absolute_path(&root) || !is_normal_absolute_path(&mount_point) {
            return Err(format!(
                "/proc/self/mountinfo line {} has a non-normal cgroup2 path",
                index + 1
            ));
        }
        if let Ok(relative) = unified_path.strip_prefix(&root) {
            candidates.push((root, mount_point.join(relative)));
        }
    }

    let Some((root, directory)) = candidates
        .into_iter()
        .max_by_key(|(root, _)| root.components().count())
    else {
        return Err(format!(
            "no cgroup2 mount contains unified membership {}",
            unified_path.display()
        ));
    };

    // Recover the selected mount point by removing the hierarchy-relative suffix.
    let relative = unified_path
        .strip_prefix(&root)
        .map_err(|_| "selected cgroup2 root no longer matches membership".to_owned())?;
    let mount_point = directory
        .ancestors()
        .nth(relative.components().count())
        .ok_or_else(|| "cgroup2 directory cannot be mapped to its mount point".to_owned())?
        .to_owned();
    Ok(Some((directory, mount_point)))
}

fn is_normal_absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && path.components().all(|component| {
            matches!(
                component,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )
        })
}

fn cgroup_ancestor_headroom(
    directory: &Path,
    mount_point: &Path,
) -> Result<Option<u64>, AvailableMemoryError> {
    if !directory.starts_with(mount_point) {
        return Err(AvailableMemoryError::Invalid(format!(
            "cgroup directory {} is outside mount point {}",
            directory.display(),
            mount_point.display()
        )));
    }

    let mut tightest = None;
    for ancestor in directory.ancestors() {
        if !ancestor.starts_with(mount_point) {
            break;
        }
        let max_path = ancestor.join("memory.max");
        let max = match fs::read_to_string(&max_path) {
            Ok(value) => value,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if ancestor == mount_point {
                    break;
                }
                continue;
            }
            Err(source) => {
                return Err(AvailableMemoryError::Read {
                    path: max_path,
                    source,
                });
            }
        };
        if max.trim() != "max" {
            let limit = max.trim().parse::<u64>().map_err(|_| {
                AvailableMemoryError::Invalid(format!(
                    "{} is neither an unsigned byte count nor max",
                    max_path.display()
                ))
            })?;
            let current_path = ancestor.join("memory.current");
            let current = read_text(&current_path)?
                .trim()
                .parse::<u64>()
                .map_err(|_| {
                    AvailableMemoryError::Invalid(format!(
                        "{} is not an unsigned byte count",
                        current_path.display()
                    ))
                })?;
            let headroom = limit.saturating_sub(current);
            tightest = Some(tightest.map_or(headroom, |bytes: u64| bytes.min(headroom)));
        }
        if ancestor == mount_point {
            break;
        }
    }
    Ok(tightest)
}

fn parse_status(status: &str) -> Result<ProcessMemory, ProcessMemoryError> {
    let resident_bytes = parse_kib_field(status, "VmRSS")?;
    let peak_resident_bytes = parse_kib_field(status, "VmHWM")?;
    if peak_resident_bytes < resident_bytes {
        return Err(ProcessMemoryError::InvalidStatus(format!(
            "VmHWM ({peak_resident_bytes} bytes) is smaller than VmRSS ({resident_bytes} bytes)"
        )));
    }
    Ok(ProcessMemory {
        resident_bytes,
        peak_resident_bytes,
    })
}

fn parse_kib_field(status: &str, field: &str) -> Result<u64, ProcessMemoryError> {
    let prefix = format!("{field}:");
    let line = status
        .lines()
        .find(|line| line.starts_with(&prefix))
        .ok_or_else(|| ProcessMemoryError::InvalidStatus(format!("missing {field}")))?;
    let mut values = line[prefix.len()..].split_whitespace();
    let value = values
        .next()
        .ok_or_else(|| ProcessMemoryError::InvalidStatus(format!("{field} has no value")))?
        .parse::<u64>()
        .map_err(|_| ProcessMemoryError::InvalidStatus(format!("invalid {field} value")))?;
    let unit = values
        .next()
        .ok_or_else(|| ProcessMemoryError::InvalidStatus(format!("{field} has no unit")))?;
    if unit != "kB" || values.next().is_some() {
        return Err(ProcessMemoryError::InvalidStatus(format!(
            "{field} must contain one kB value"
        )));
    }
    value
        .checked_mul(1024)
        .ok_or_else(|| ProcessMemoryError::InvalidStatus(format!("{field} byte count overflows")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_residency_in_bytes() {
        let memory = parse_status("Name:\ttest\nVmHWM:\t  4096 kB\nVmRSS:\t2048 kB\n").unwrap();
        assert_eq!(memory.resident_bytes(), 2 * 1024 * 1024);
        assert_eq!(memory.peak_resident_bytes(), 4 * 1024 * 1024);
    }

    #[test]
    fn rejects_missing_units_overflow_and_impossible_peaks() {
        assert!(parse_status("VmHWM: 2 kB\nVmRSS: 3 kB\n").is_err());
        assert!(parse_status("VmHWM: 3 kB\nVmRSS: 2 bytes\n").is_err());
        assert!(parse_status("VmHWM: 3 kB\n").is_err());
        assert!(parse_status(&format!("VmHWM: {} kB\nVmRSS: 0 kB\n", u64::MAX)).is_err());
    }

    #[test]
    fn live_process_measurement_is_ordered() {
        let memory = process_memory().unwrap();
        assert!(memory.resident_bytes() > 0);
        assert!(memory.peak_resident_bytes() >= memory.resident_bytes());
    }

    #[test]
    fn parses_cgroup_byte_counts_without_accepting_whitespace_or_sentinels() {
        let path = Path::new("/sys/fs/cgroup/example/memory.current");
        assert_eq!(parse_cgroup_byte_count("0\n", path).unwrap(), 0);
        assert_eq!(
            parse_cgroup_byte_count(&u64::MAX.to_string(), path).unwrap(),
            u64::MAX
        );

        for invalid in [
            "",
            "\n",
            " 1\n",
            "1 \n",
            "+1\n",
            "-1\n",
            "max\n",
            "1\n2\n",
            "1\n\n",
            "18446744073709551616\n",
        ] {
            assert!(
                parse_cgroup_byte_count(invalid, path).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn samples_strict_cgroup_current_and_peak_counters() {
        let directory =
            std::env::temp_dir().join(format!("beholdd-cgroup-sample-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("memory.current"), "321\n").unwrap();
        fs::write(directory.join("memory.peak"), "654\n").unwrap();

        let memory = sample_cgroup_memory(&directory).unwrap();
        assert_eq!(memory.current_bytes(), 321);
        assert_eq!(memory.peak_bytes(), 654);

        fs::write(directory.join("memory.peak"), "320\n").unwrap();
        assert!(sample_cgroup_memory(&directory).is_err());
        fs::write(directory.join("memory.peak"), "654 kB\n").unwrap();
        assert!(sample_cgroup_memory(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cgroup_sampling_distinguishes_absent_v2_from_v1_or_hybrid_memory() {
        assert_eq!(cgroup_memory_from("2:cpu:/old\n", "").unwrap(), None);
        assert!(cgroup_memory_from("2:memory:/old\n", "").is_err());
        assert!(cgroup_memory_from("0::/v2\n2:memory:/v1\n", "").is_err());
    }

    #[test]
    fn live_cgroup_measurement_follows_the_resolved_leaf_when_supported() {
        let membership = fs::read_to_string("/proc/self/cgroup").unwrap();
        let mountinfo = fs::read_to_string("/proc/self/mountinfo").unwrap();
        match resolve_cgroup2_directory(&membership, &mountinfo) {
            Err(_) => assert!(cgroup_memory().is_err()),
            Ok(None) => assert_eq!(cgroup_memory().unwrap(), None),
            Ok(Some((directory, _)))
                if directory.join("memory.current").is_file()
                    && directory.join("memory.peak").is_file() =>
            {
                let memory = cgroup_memory().unwrap().unwrap();
                assert!(memory.peak_bytes() >= memory.current_bytes());
            }
            Ok(Some(_)) => {
                assert!(matches!(
                    cgroup_memory(),
                    Err(CgroupMemoryError::Read { .. })
                ));
            }
        }
    }

    #[test]
    fn parses_host_availability_without_guessing_units() {
        assert_eq!(
            parse_available_memory("MemTotal: 10 kB\nMemAvailable: 4096 kB\n").unwrap(),
            4 * 1024 * 1024
        );
        assert!(parse_available_memory("MemAvailable: 4 MB\n").is_err());
        assert!(parse_available_memory("MemTotal: 4 kB\n").is_err());
        assert!(parse_available_memory(&format!("MemAvailable: {} kB\n", u64::MAX)).is_err());
    }

    #[test]
    fn resolves_a_non_root_unified_cgroup_through_its_mount_root() {
        let membership = "0::/tenant/user.slice/job.scope\n";
        let mountinfo =
            "31 20 0:29 /tenant /sys/fs/cgroup rw - cgroup2 cgroup2 rw,memory_recursiveprot\n";
        let (directory, mount_point) = resolve_cgroup2_directory(membership, mountinfo)
            .unwrap()
            .unwrap();
        assert_eq!(directory, Path::new("/sys/fs/cgroup/user.slice/job.scope"));
        assert_eq!(mount_point, Path::new("/sys/fs/cgroup"));
    }

    #[test]
    fn cgroup_resolution_is_conservative_about_ambiguous_or_unsafe_paths() {
        assert!(resolve_cgroup2_directory("2:memory:/old\n", "").is_err());
        assert!(resolve_cgroup2_directory("0::/v2\n2:memory:/v1\n", "").is_err());
        assert_eq!(resolve_cgroup2_directory("2:cpu:/old\n", "").unwrap(), None);
        assert!(resolve_cgroup2_directory("0::/a\n0::/b\n", "").is_err());
        assert!(resolve_cgroup2_directory("malformed\n", "").is_err());
        assert!(resolve_cgroup2_directory(
            "0::/tenant/job\n",
            "31 20 0:29 /elsewhere /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"
        )
        .is_err());
        assert!(resolve_cgroup2_directory(
            "0::/tenant/../escape\n",
            "31 20 0:29 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n"
        )
        .is_err());
    }

    #[test]
    fn live_availability_is_positive_and_never_exceeds_the_host() {
        let memory = available_memory().unwrap();
        assert!(memory.host_available_bytes() > 0);
        assert!(memory.effective_available_bytes() <= memory.host_available_bytes());
        if let Some(cgroup) = memory.cgroup_available_bytes() {
            assert!(memory.effective_available_bytes() <= cgroup);
        }
    }

    #[test]
    fn finite_cgroup_ancestor_headroom_uses_the_tightest_hard_limit() {
        let root =
            std::env::temp_dir().join(format!("beholdd-cgroup-memory-{}", std::process::id()));
        let child = root.join("user.slice");
        let leaf = child.join("job.scope");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&leaf).unwrap();
        fs::write(root.join("memory.max"), "100\n").unwrap();
        fs::write(root.join("memory.current"), "20\n").unwrap();
        fs::write(child.join("memory.max"), "80\n").unwrap();
        fs::write(child.join("memory.current"), "50\n").unwrap();
        fs::write(leaf.join("memory.max"), "max\n").unwrap();

        assert_eq!(cgroup_ancestor_headroom(&leaf, &root).unwrap(), Some(30));

        fs::write(child.join("memory.current"), "81\n").unwrap();
        assert_eq!(cgroup_ancestor_headroom(&leaf, &root).unwrap(), Some(0));
        fs::remove_dir_all(root).unwrap();
    }
}
