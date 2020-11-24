rem git remote add upstream https://github.com/OSGeo/grass-addons.git
rem git fetch upstream

git checkout master
git merge upstream/master
git push origin
git commit -a -m "Updates to repo %date%"
git push origin