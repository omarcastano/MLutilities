import pytest
from mlutilities.eda import kolmogorov_test
import numpy as np


dict_dataset = {
    "normal_dist": np.random.normal(0, 1, 1000), 
    "uniform_dist": np.random.uniform(-1, 1, 1000),
    "cat_var": np.random.choice(["a", "b", "c"], size=1000)
    }


class TestKolmogorov:
    def test_kolmogorov_normal_dist(self):
        result = kolmogorov_test(dict_dataset, variable="normal_dist", return_test=True, print_test=True)

        assert result[1] > 0.05
        assert result[2] == "Normal distribution"

    def test_kolmogorov_no_normal_dist(self):
        result = kolmogorov_test(dict_dataset, variable="uniform_dist", return_test=True, print_test=True)

        assert result[1] < 0.05
        assert result[2] == "Not normal distribution"

    @pytest.mark.parametrize("transformation", ["log", "yeo_johnson"])
    def test_kolmogorov_with_transformation(self, transformation):
        results = kolmogorov_test(dict_dataset, variable="uniform_dist", transformation=transformation, return_test=True, print_test=True)

        assert results[1] < 0.05
        assert results[2] == "Not normal distribution"

    def test_kolmogorov_whit_histogram_plot(self):
        # check if plotting does not rise and error
        try:
            kolmogorov_test(dict_dataset, variable="normal_dist", plot_histogram=True, print_test=True)
        except:
            pytest.fail("Plotting failed")

    def test_kolmogorov_whit_printing_test(self):
        try:
            kolmogorov_test(dict_dataset, variable="normal_dist", print_test=True)
        except:
            pytest.fail("Printing failed")

    def test_kolmogorov_with_color_argument(self):
        try:
            kolmogorov_test(dict_dataset, variable="normal_dist", color="cat_var", print_test=False, plot_histogram=True)
        except:
            pytest.fail("Printing failed when using color argument")
        

        
