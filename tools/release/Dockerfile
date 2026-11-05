# Official linux/amd64 image digests verified against registry.docker.io.
# Every layer uses Debian bullseye (glibc 2.31); no package-manager refresh occurs.
FROM python:3.12.10-bullseye@sha256:f999fda265134523a8a65089edf0c1d94371c800639844030648f77fabd29493 AS python
FROM node:22.20.0-bullseye@sha256:9d81853ab1fcc62dc9da96155946942d598ef13b3f887b216fcee3963edf6e90 AS node
FROM rust:1.98.0-bullseye@sha256:58643a517df55089a44540d827eb9bfd8d5f121c249767be6e25f24b68c0563d
COPY --from=python /usr/local/bin/python3.12 /usr/local/bin/python3
COPY --from=python /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/libpython3.12.so.1.0
COPY --from=python /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=node /usr/local/bin/node /usr/local/bin/node
ENV LD_LIBRARY_PATH=/usr/local/lib
