import numpy as np
import matplotlib.pyplot as plt

# Define both paths
path1 = "synthe_py/out/test_fixed_5770.spec"
path2 = "grids/at12_aaaaa/spec/at12_aaaaa_t05770g4.44.spec"
# path2 = "grids/at12_aaaaa/spec/at12_aaaaa_t02500g-1.0.spec"
# path2 = "grids/at12_aaaaa/spec/at12_aaaaa_t03750g3.50.spec"

# Load both spectra
data1 = np.loadtxt(path1)
data2 = np.loadtxt(path2)

# Extract columns for both
wavelength1, flux1, continuum1 = data1[:, 0], data1[:, 1], data1[:, 2]
wavelength2, flux2, continuum2 = data2[:, 0], data2[:, 1], data2[:, 2]

normalized_flux1 = flux1 / continuum1
normalized_flux2 = flux2 / continuum2

# Print 10 wavelengths where flux > continuum for spectrum 1
mask1 = flux1 > continuum1
wavelengths_flux_gt_continuum1 = wavelength1[mask1]
print("First 10 wavelengths in", path1, "where flux > continuum:")
print(wavelengths_flux_gt_continuum1[:10])

# Print 10 wavelengths where flux > continuum for spectrum 2
mask2 = flux2 > continuum2
wavelengths_flux_gt_continuum2 = wavelength2[mask2]
print("First 10 wavelengths in", path2, "where flux > continuum:")
print(wavelengths_flux_gt_continuum2[:10])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# Set y-limits only from finite values (to avoid NaN/Inf errors)
# But plot all points (matplotlib will skip NaN/Inf automatically)
finite_mask1 = np.isfinite(normalized_flux1)
finite_mask2 = np.isfinite(normalized_flux2)

if np.sum(finite_mask1) > 0:
    finite_flux1 = normalized_flux1[finite_mask1]
    margin1 = 0.05 * (np.max(finite_flux1) - np.min(finite_flux1))
    axes[0].set_ylim(np.min(finite_flux1) - margin1, np.max(finite_flux1) + margin1)

if np.sum(finite_mask2) > 0:
    finite_flux2 = normalized_flux2[finite_mask2]
    margin2 = 0.05 * (np.max(finite_flux2) - np.min(finite_flux2))
    axes[1].set_ylim(np.min(finite_flux2) - margin2, np.max(finite_flux2) + margin2)

# First spectrum - plot all points
axes[0].plot(wavelength1, normalized_flux1, label="Normalized Flux")
axes[0].axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Continuum")
axes[0].set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Normalized Flux")
axes[0].set_title(f"Normalized Spectrum\n{path1}")
axes[0].legend()
count1 = np.sum(mask1)
axes[0].text(
    0.98,
    0.95,
    f"Flux > continuum: {count1} / {len(flux1)}\n({100*count1/len(flux1):.2f}%)",
    ha="right",
    va="top",
    transform=axes[0].transAxes,
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
)

# Second spectrum - plot all points
axes[1].plot(wavelength2, normalized_flux2, label="Normalized Flux")
axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Continuum")
axes[1].set_xlabel("Wavelength (nm)")
axes[1].set_title(f"Normalized Spectrum\n{path2}")
axes[1].legend()
count2 = np.sum(mask2)
axes[1].text(
    0.98,
    0.95,
    f"Flux > continuum: {count2} / {len(flux2)}\n({100*count2/len(flux2):.2f}%)",
    ha="right",
    va="top",
    transform=axes[1].transAxes,
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
)

plt.suptitle("Side-by-side Normalized Spectra Comparison")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("normalized_spectrum_side_by_side.png", dpi=300, bbox_inches="tight")
plt.show()
