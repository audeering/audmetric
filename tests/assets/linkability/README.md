To calculate the reference values run:

```bash
$ git clone https://gitlab.inria.fr/magnet/anonymization_metrics.git
$ cd anonymization_metrics
$ virtualenv --python=python3.8 env
$ source env/bin/activate
$ pip install -r requirements.txt
$ cd ../../../../
$ pip install -r requirements.txt
$ cd -
$ cd ..
$ python linkability_reference.py
```
