@echo off

call conda update conda -y
call conda update --all -y
call conda env create -f cv3.yml
call conda activate cvCoursework2020