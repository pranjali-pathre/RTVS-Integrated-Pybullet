from scipy.spatial.transform import Rotation as R

r = R.from_matrix([[-1.55678691e-01,  4.44531940e-02, -9.86806953e-01],
                            [ 9.87807682e-01,  7.00583208e-03, -1.55520969e-01],
                            [-3.35068258e-10, -9.98986786e-01, -4.50018705e-02]]
)
print("r: ", r.as_euler("xyz", degrees=True))