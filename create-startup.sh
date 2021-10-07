#!/bin/bash

echo "#!/bin/bash" > $2/startup-run
# echo "dji" | sudo -S ./jetson_clocks.sh
# gnome-terminal -x ./tegrastats
#echo "echo dji | sudo -S sudo ./jetson_clocks.sh" >> $2/startup-run
echo "gnome-terminal -- bash -c \"echo dji | sudo -S $1/monitor.sh \\\"$2/RM2021_Adv\\\"\"" >> $2/startup-run
chmod +x $2/startup-run

