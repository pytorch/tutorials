#!/bin/bash
# Based off of https://github.com/pytorch/pytorch/tree/b52e0bf131a4e55cd987176f9c5a8d2ad6783b4f/.ci/docker
set -ex

# Mirror jenkins user in container
# jenkins user as ec2-user should have the same user-id
echo "jenkins:x:1000:1000::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:1000:" >> /etc/group
# Needed on focal or newer
echo "jenkins:*:19110:0:99999:7:::" >>/etc/shadow

# Create $HOME
mkdir -p /var/lib/jenkins
chown jenkins:jenkins /var/lib/jenkins
mkdir -p /var/lib/jenkins/.ccache
chown jenkins:jenkins /var/lib/jenkins/.ccache

# Allow writing to /usr/local (for make install)
chown jenkins:jenkins /usr/local

# Allow sudo
# TODO: Maybe we shouldn't
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins

# Test that sudo works
sudo -u jenkins sudo -v
