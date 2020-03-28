# projector
`projector` contains code to numerically calculate excess-surface-density (ESD) profiles from radial density profiles.

Specifically, given some rho(r), you can use `projector` to numerically estimate DeltaSigma(r), a quantity which measures the difference between the average projected mass density within a circle, and the value at the boundary. This quantity is used in weak-lensing analysis of galaxy cluster masses.

`projector` uses some rather crude numerical integration techniques, but is able to get accuracy to within 1% for NFW profiles, for which the ESD profiles are known analytically.

# usage
To calculate an ESD, install and import `projector`. You can then use `projector.esd` to calculate an ESD: `projector.esd` requires the radii at which you want to evaluate the ESD, and a density function which can take radii inputs as 1 or 2D numpy arrays. It also allows you to specify how many points are used to numerically estimate the projection integral. The default of 120 is a decent compromise between performance and accuracy, at least for the cases I have tested.

# example
```python
radii = np.linspace(0.1, 10, 10)
rho_func = lambda r: 1/r
esds = projector.esd(radii, rho_func)
```
For this density function, the ESD should be a constant equal to 1 for all radii. You can find an automated test of this in the `spec` directory.

# to-do
This project needs better documentation. Once the research project using this code is finished, I will create a release version of this library with good docs. For now this project is in a WIP/beta state.
