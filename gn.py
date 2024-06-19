import pytest
from pytest import approx
import os
from os import path as pth
import numpy as np
from astropy.io import fits
import scopesim
from scopesim import rc
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

if rc.__config__["!SIM.tests.run_integration_tests"] is False:
    pytestmark = pytest.mark.skip("Ignoring DREAMS integration tests")

TOP_PATH = pth.abspath(pth.join(pth.dirname(__file__), "../../"))
rc.__config__["!SIM.file.local_packages_path"] = TOP_PATH

PKGS = {"DREAMS": "/Users/anjali/Desktop/DREAMS.zip"}

PLOTS = True

class TestInit:
    def test_all_packages_are_available(self):
        rc_local_path = rc.__config__["!SIM.file.local_packages_path"]
        for pkg_name in PKGS:
            assert os.path.isdir(os.path.join(rc_local_path, pkg_name))
        print("irdb" not in rc_local_path)

class TestLoadUserCommands:
    def test_user_commands_loads_without_throwing_errors(self, capsys):
        cmd = scopesim.UserCommands(use_instrument="DREAMS")
        assert isinstance(cmd, scopesim.UserCommands)

        stdout = capsys.readouterr()
        assert len(stdout.out) == 0

class TestMakeOpticalTrain:
    def __init__(self):
        self.hdu_list = None

    def test_load_lfao(self):
        cmd = scopesim.UserCommands(use_instrument="DREAMS",
                                    properties={"!OBS.filter_name": "I",
                                                "!OBS.dit": 10,
                                                "!DET.bin_size": 1,
                                                "!OBS.sky.bg_mag": 14.9,
                                                "!OBS.sky.filter_name": "I"})
        opt = scopesim.OpticalTrain(cmd)
        opt["detector_linearity"].include = False
        assert isinstance(opt, scopesim.OpticalTrain)

        src = scopesim.source.source_templates.star_field(10000, 10, 20, 700)
        opt.observe(src)
        self.hdu_list = opt.readout()[0]

        assert isinstance(self.hdu_list, fits.HDUList)

        print(np.average(self.hdu_list[1].data))
        if PLOTS:
            self.plot_data()

    def plot_data(self):
        if self.hdu_list is not None:
            for my_hdu in self.hdu_list[1:]:
                plt.imshow(my_hdu.data, norm=LogNorm())
                plt.colorbar()  # Add a colorbar to the plot
                plt.title("Observed Star Field")  # Add a title to the plot
                plt.xlabel("X Pixels")  # Add an x-axis label
                plt.ylabel("Y Pixels")  # Add a y-axis label
                plt.show()

def run_test_and_plot():
    test_optical_train = TestMakeOpticalTrain()
    test_optical_train.test_load_lfao()
    # test_optical_train.plot_data()

# Run the test and plot as soon as the module is imported
run_test_and_plot()
