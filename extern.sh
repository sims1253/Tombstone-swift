#
# connect.sh
# 
# Copyright 2014 Daniel Jährig <daniel.jaehrig@tuhh.de>
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
# 


#!/bin/bash
#$1 command
#$2 computer
#$3 RZ-Login
#$4 studX
#$5 mountpos

if [ $# -ne 5 ]
	then
		echo -e "missing arguments!\nextern.sh command computer RZ-Login studX folder\n\ncommand:\n\ttunnel\t: ssh tunnel to TUHH\n\tmount\t: mount xy on local\n\tssh\t: establish ssh connection to xy\n\tall\t: ssh tunnel, mount and ssh connection\n\tunmount\t: unmount xy (requires superuser rights)\n\ncomputer:\n\txy3 : AMD\n\txy4 : XEON Phi\n\txy5 : NVIDIA\n\nRZ-Login: your RZ login (eg. sabc1234)\n\nstudX: your group login (eg. stud1)\n\nfolder: folder to (un)mount (direct or relative path)"
	exit
fi

case $1 in
        "tunnel") ssh -N -f -L 2126:$2.ti6.tu-harburg.de:22 $3@ssh.rz.tu-harburg.de
            ;;
        "mount") echo -n "$4's "
				 sshfs $4@localhost:/mounts/student/$4 $5 -p 2126 -o workaround=all -o TCPKeepAlive=yes
            ;;
        "ssh") ssh -p 2126 $4@localhost
            ;;
        "all") ssh -N -f -L 2126:$2.ti6.tu-harburg.de:22 $3@ssh.rz.tu-harburg.de && echo -n "$4's " && sshfs $4@localhost:/mounts/student/$4 $5 -p 2126 -o workaround=all -o TCPKeepAlive=yes && ssh -p 2126 $4@localhost
            ;;
        "unmount") sudo umount -f $5
            ;;
        *) echo -e "extern.sh command computer RZ-Login studX folder\n\ncommand:\n\ttunnel\t: ssh tunnel to TUHH\n\tmount\t: mount xy on local\n\tssh\t: establish ssh connection to xy\n\tall\t: ssh tunnel, mount and ssh connection\n\tunmount\t: unmount xy (requires superuser rights)\n\ncomputer:\n\txy3 : AMD\n\txy4 : XEON Phi\n\txy5 : NVIDIA\n\nRZ-Login: your RZ login (eg. sabc1234)\n\nstudX: your group login (eg. stud1)\n\nfolder: folder to (un)mount (direct or relative path)" 
            ;;
esac
