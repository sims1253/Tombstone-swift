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
#$3 studX
#$4 mountpos

if [ $# -ne 4 ]
	then
		echo -e "missing arguments!\nintern.sh command computer studX folder\n\ncommand:\n\tmount\t: mount xy on local\n\tssh\t: establish ssh connection to xy\n\tall\t: mount and ssh connection\n\tunmount\t: unmount xy (requires superuser rights)\n\ncomputer:\n\txy3 : AMD\n\txy4 : XEON Phi\n\txy5 : NVIDIA\n\nstudX: your group login (eg. stud1)\n\nfolder: folder to (un)mount (direct or relative path)"
	exit
fi

case $1 in
        "mount") echo -n "$3's "
				 sshfs $3@$2.ti6.tu-harburg.de:/mounts/student/$3 $4 -o workaround=all -o TCPKeepAlive=yes
            ;;
        "ssh") echo -n "$3's " && ssh $3@$2.ti6.tu-harburg.de
            ;;
        "all") echo -n "$3's " && sshfs $3@$2.ti6.tu-harburg.de:/mounts/student/$3 $4 -o workaround=all -o TCPKeepAlive=yes && echo -n "$3's " && ssh $3@$2.ti6.tu-harburg.de
            ;;
        "unmount") sudo umount -f $4
            ;;
        *) echo -e "intern.sh command computer  studX folder\n\ncommand:\n\tmount\t: mount xy on local\n\tssh\t: establish ssh connection to xy\n\tall\t: mount and ssh connection\n\tunmount\t: unmount xy (requires superuser rights)\n\ncomputer:\n\txy3 : AMD\n\txy4 : XEON Phi\n\txy5 : NVIDIA\n\nstudX: your group login (eg. stud1)\n\nfolder: folder to (un)mount (direct or relative path)" 
            ;;
esac
