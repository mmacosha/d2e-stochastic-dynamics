#!/bin/bash
FROM='/Users/ktamogashev/Documents/PhD/projects/sb'
TO='/home/s2754864/repos/energy-sb'

rsync --exclude-from="exclude.txt" -avzP $FROM mlp:$TO
