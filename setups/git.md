# Git commands helper

### Add a new repository

1. Fork an interesting repo 
2. Add it in the git of the benchmarking as a submodule 
```
git submodule add <url> a/submodule 
```
3. Create a script to adapt the method on our dataset

### To clone this repo on a new machine

Option 1:
`git clone --recurse-submodules ...`

Option 2:
1. `git clone ...`
2. `git submodule update --init`

Option 3:
1. `git clone ...`
2. `git submodule init`
3. `git submodule update` to fetch changes from the submodule

### git pull for submodules

1. `git pull`
2. `git submodule update --init --recursive`

### push submodules

`git push --recurse-submodules=check`


