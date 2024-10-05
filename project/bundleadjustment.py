'''
Sample Usage:-
python bundleadjustment.py --K_Matrix K_matrix.npy --D_Coeff distortion.npy --type DICT_5X5_100
'''

from cameraPose import *
from utils import ARUCO_DICT
from scipy.optimize import least_squares

all_errors_before_adjustment = []
all_errors_after_adjustment = []
def bundle_adjustment(marker_poses, camera_matrix, dist_coeffs):
    obj_points = np.array([[0, 0, 0], 
                           [0.048, 0, 0], 
                           [0.048, 0.048, 0], 
                           [0, 0.048, 0]], dtype=np.float32) # Defines the 3D coordinates of the corners of an ArUco marker in the marker's local coordinate system. 
    if not marker_poses:
        print("No markers available for bundle adjustment.")
        return None

    print(f"Processing {len(marker_poses)} markers for bundle adjustment.")
    # Concatenates the initial guesses for the rotation vectors (rvec) and translation vectors (tvec) of all markers into a single flat array. This array will be used as the initial guess for the optimization process.
    initial_params = np.concatenate([np.ravel(node.rvec) for node in marker_poses] +
                                    [np.ravel(node.tvec) for node in marker_poses])

    def reprojection_error(params):
        '''This function computes the reprojection error, which measures the difference between the observed 2D 
        marker corners in the image and the 2D points projected from the 3D model. It calculates this error for 
        each marker pose.'''
        num_markers = len(marker_poses)
        errors = []
        for i in range(num_markers):
            rvec = params[3*i:3*i+3] # extracts the rotation vector for the i-th marker pose from the flat array.
            tvec = params[3*num_markers+3*i:3*num_markers+3*i+3] # extracts the translation vector for the i-th marker pose from the flat array.
            projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
            observed_points = marker_poses[i].corners.reshape(-1, 2)
            errors.append(np.linalg.norm(projected_points.squeeze() - observed_points, axis=1))
        total_errors = np.concatenate(errors)
        # print("Total errors", total_errors)

        return total_errors   


    error_before = reprojection_error(initial_params)
    global all_errors_before_adjustment
    all_errors_before_adjustment.extend(error_before[:50])
    # Minimizes the reprojection error using the least_squares optimization method.
    result = least_squares(reprojection_error, initial_params, verbose=1, xtol=1e-3, ftol=1e-3, gtol=1e-3)
    # Reshape the result to separate rvec and tvec for each marker
    num_markers = len(marker_poses)
    rvecs = result.x[:3*num_markers].reshape(num_markers, 3)
    tvecs = result.x[3*num_markers:].reshape(num_markers, 3)
    global all_errors_after_adjustment
    errors_after = reprojection_error(result.x)
    all_errors_after_adjustment.extend(errors_after[:50])
    return rvecs, tvecs

def plot_errors(errors_before, errors_after):
    labels = ['Before Adjustment', 'After Adjustment']
    data = [errors_before, errors_after]
    plt.boxplot(data, labels=labels)
    plt.title('Reprojection Errors Before and After Adjustment')
    plt.ylabel('Reprojection Error')
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUco tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUco tag type '{args['type']}' is not supported")
        sys.exit(0)

    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])
    aruco_dict_type = ARUCO_DICT[args["type"]]
    cumulative_rvec = np.zeros((3, 1))
    cumulative_tvec = np.zeros((3, 1))
    count = 0

    video = cv2.VideoCapture('pose.MOV')
    all_marker_poses = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        _, marker_poses= pose_estimation(frame, aruco_dict_type, k, d)
        if marker_poses:
            all_marker_poses.extend(marker_poses)
        else:
            print("No markers detected in frame.")

    video.release()

    # Bundle Adjustment Step
    if all_marker_poses:
        rvecs, tvecs = bundle_adjustment(all_marker_poses, k, d)
        for i in range(len(rvecs)):
            R_inv, tvec_inv = invert_pose(rvecs[i], tvecs[i])
            cumulative_rvec += R_inv
            cumulative_tvec += tvec_inv
            count += 1
            # print(f"Camera Pose {i}: Inverted Rotation Matrix = \n{R_inv}")
            # print(f"Inverted Translation Vector = {tvec_inv}")
    else:
        print("No markers were detected across all frames.")
    if count > 1:
        average_rvec = cumulative_rvec / count
        average_tvec = cumulative_tvec / count
        print("Average Camera Pose (Rotation Vector):", average_rvec)
        print("Average Camera Pose (Translation Vector):", average_tvec)
    plot_errors(all_errors_before_adjustment, all_errors_after_adjustment)
    # print("All error",all_errors_after_adjustment)
    # print("All error after",all_errors_before_adjustment)

if __name__ == '__main__':
    main()
