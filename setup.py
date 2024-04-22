from setuptools import find_packages, setup

package_name = 'activity_zed'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ns',
    maintainer_email='noushad.sust@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_sub = activity_zed.zed_sub:main',
            'zed_inf = activity_zed.zed_inf:main'
            'zed_inf_cpu = activity_zed.zed_inf_cpu:main'
        ],
    },
)
