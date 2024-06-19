# integration test using everything and the DREAMS package
import pytest
from pytest import approx
import os
from os import path as pth
import shutil

import numpy as np
from astropy import units as u
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

PLOTS = False


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
    def test_load_lfao(self):
        cmd = scopesim.UserCommands(use_instrument="DREAMS",
                                    properties={"!OBS.filter_name": "J",
                                                "!OBS.dit": 10,
                                                "!DET.bin_size": 1,
                                                "!OBS.sky.bg_mag": 14.9,
                                                "!OBS.sky.filter_name": "J"})
        opt = scopesim.OpticalTrain(cmd)
        opt["detector_linearity"].include = False
        assert isinstance(opt, scopesim.OpticalTrain)

        src = scopesim.source.source_templates.star_field(10000, 10, 20, 700)
        # src = scopesim.source.source_templates.empty_sky()
        opt.observe(src)
        hdu_list = opt.readout()[0]

        assert isinstance(hdu_list, fits.HDUList)

        print(np.average(hdu_list[1].data))
        if PLOTS:
            plt.imshow(hdu_list[1].data, norm=LogNorm())
            plt.show()
