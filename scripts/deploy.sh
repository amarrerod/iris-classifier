#!\bin\bash

coverage html
mv htmlcov/* docs
rmdir htmlcov