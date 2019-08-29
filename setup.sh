#!/bin/bash
apt-get update
apt-get install firewalld vim
apt-get upgrade
firewall-cmd --remove-service=ssh --permanent
firewall-cmd --reload
passwd -l root
apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
 apt-get update
 apt-get install docker-ce docker-ce-cli containerd.io
 docker run hello-world
