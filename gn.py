import os
import pytest
import numpy as np
from astropy.io.fits import HDUList
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scopesim

from scopesim import rc
from scopesim.source.source_templates import star_field
import scopesim_templates as sim_tp

PLOTS = False

if rc.__config__["!SIM.tests.run_integration_tests"] is False:
    pytestmark = pytest.mark.skip("Ignoring DREAMS integration tests")

# Set TOP_PATH to the directory containing the DREAMS package
TOP_PATH = "/Users/anjali/Desktop"
rc.__config__["!SIM.file.local_packages_path"] = TOP_PATH

# Adjust the PKGS dictionary to reflect the correct path
PKGS = {"DREAMS": os.path.join(TOP_PATH, "DREAMS")}

# Verify the path to the DREAMS package
if not os.path.exists(PKGS["DREAMS"]):
    raise FileNotFoundError(f"DREAMS package not found at {PKGS['DREAMS']}")
else:
    print("DREAMS package found at:", PKGS["DREAMS"])

class TestLoads:
    def test_scopesim_loads_package(self):
        dreams = scopesim.OpticalTrain("DREAMS")
        assert isinstance(dreams, scopesim.OpticalTrain)  # Corrected syntax
        print("scopesim package loaded successfully.")

class TestObserves:
    def test_something_comes_out(self):
        print("Starting observation test...")
        src = star_field(10000, 10, 20, width=700)

        cmds = scopesim.UserCommands(use_instrument="DREAMS")
        cmds["!OBS.dit"] = 10
        cmds["!DET.bin_size"] = 1
        cmds["!OBS.sky.bg_mag"] = 14.9
        cmds["!OBS.sky.filter_name"] = "J"

        dreams = scopesim.OpticalTrain(cmds)
        dreams["detector_linearity"].include = False
        dreams.observe(src)
        hdus = dreams.readout()

        print(f"Observation completed. HDUList type: {type(hdus[0])}")

        if PLOTS:
            plt.subplot(121)
            wave = np.arange(3000, 11000)
            plt.plot(wave, dreams.optics_manager.surfaces_table.throughput(wave))

            plt.subplot(122)
            im = hdus[0][1].data
            plt.imshow(im, norm=LogNorm())
            plt.colorbar()
            plt.title("Observed Star Field")
            plt.xlabel("X Pixels")
            plt.ylabel("Y Pixels")
            plt.show()

    @pytest.mark.slow
    def test_observes_from_scopesim_templates(self):
        print("Starting scopesim templates observation test...")
        src = sim_tp.stellar.cluster(mass=10000, distance=2000, core_radius=1)

        dreams = scopesim.OpticalTrain("DREAMS")
        dreams.observe(src)

        dreams.cmds["!OBS.dit"] = 10
        hdus = dreams.readout()

        assert isinstance(hdus[0], HDUList)
        print("Observation from scopesim templates completed.")

        if PLOTS:
            im = hdus[0][1].data
            plt.imshow(im, norm=LogNorm(), cmap="hot")
            plt.colorbar()
            plt.show()

    @pytest.mark.slow
    def test_saves_readout_to_disc(self):
        print("Starting test to save readout to disk...")
        src = sim_tp.stellar.cluster(mass=10000, distance=2000, core_radius=1)
        dreams = scopesim.OpticalTrain("DREAMS")
        dreams.observe(src)
        dreams.readout(filename="GNANU.fits")

        assert os.path.exists("GNANU.fits")
        print("Readout saved to GNANU.fits.")

def run_test_and_plot():
    test_observes = TestObserves()
    test_observes.test_something_comes_out()

# Run the test and plot as soon as the module is imported
run_test_and_plot()

