from setuptools import setup

from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


def custom_command():
    from subprocess import check_output

    folder = 'network_dismantling/common/external_dismantlers/'
    cd_cmd = 'cd {} && '.format(folder)
    cmd = 'make clean && make'

    try:
        print(check_output(cd_cmd + cmd, shell=True, text=True))
    except Exception as e:
        print("ERROR! {}".format(e))

        exit(-1)


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        custom_command()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        custom_command()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        custom_command()


setup(
    name='NDML',
    version='1.0',
    packages=['network_dismantling', 'network_dismantling.common', 'network_dismantling.common.external_dismantlers',
              'network_dismantling.machine_learning', 'network_dismantling.machine_learning.pytorch',
              'network_dismantling.machine_learning.pytorch.models',
              'network_dismantling.machine_learning.pytorch.reinsertion'],
    url='',
    license='',
    author='Marco Grassia',
    author_email='',
    description='',
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'egg_info': CustomEggInfoCommand,
    }
)
