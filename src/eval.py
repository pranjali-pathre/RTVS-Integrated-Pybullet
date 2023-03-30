import numpy as np
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, name):
        self.name = name
        self.iou_error = []
        self.trajectory_length = 0
        self.time_taken = 0
        self.photometric_error = []  # useless for now
        self.obj_speed = 0


class Evaluate:
    def __init__(self, file):
        self.file = file
        self.data = np.load(file, allow_pickle=True)
        self.data = self.data["results"].item()
        self.ct_s = list(self.data.keys())
        self.ct_s.remove("gt")

        self.metrics = {"rtvs": Metrics("rtvs"), "ours": Metrics("ours")}

    def set_iou_errors(self, ct):
        self.metrics[ct].iou_error = self.data[ct]["err"]

    def set_trajectory_length(self, ct):
        tr_len = 0
        ee_poses = self.data[ct]["ee_pos"]
        for i in range(len(ee_poses) - 1):
            tr_len += np.linalg.norm(ee_poses[i] - ee_poses[i + 1])
        self.metrics[ct].trajectory_length = round(tr_len, 4)

    def set_time_taken(self, ct):
        self.metrics[ct].time_taken = self.data[ct]["grasp_time"]

    def set_photometric_error(self, ct):
        pass

    def set_obj_speed(self, ct):
        speeds = []
        obj_poses = self.data[ct]["obj_pos"]
        for i in range(len(obj_poses) - 1):
            speeds.append(np.linalg.norm(obj_poses[i] - obj_poses[i + 1]))
        self.metrics[ct].obj_speed = round(np.mean(speeds), 4)

    def set_metrics(self):
        for ct in self.ct_s:
            self.set_iou_errors(ct)
            self.set_trajectory_length(ct)
            self.set_time_taken(ct)
            self.set_photometric_error(ct)
            self.set_obj_speed(ct)

    def gen_plots(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for ct in self.ct_s:
            ax[0].plot(self.data[ct]["t"], self.metrics[ct].iou_error, label=ct)
        ax[0].legend()
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("IoU")
        ax[0].set_title("IoU vs Time")
        ax[0].axvline(x=self.data["gt"]["grasp_time"], color="red", linestyle="--")
        ax[0].grid()

        ax[1].axis("off")
        ax[1].axis("tight")
        ax[1].text(0, 0.6, f"Object Motion: {self.data['gt']['obj_motion']}")
        # ax[1].table(
        #     cellText=[
        #         [
        #             self.metrics[ct].trajectory_length,
        #             self.metrics[ct].time_taken,
        #             self.metrics[ct].obj_speed,
        #         ]
        #         for ct in self.ct_s
        #     ],
        #     colLabels=["Trajectory Length", "Time Taken", "Object Speed"],
        #     rowLabels=self.ct_s,
        #     loc="center",
        # )
        plt.savefig("plot.png")


def main():
    file = "../results/data.npz"
    eval = Evaluate(file)
    eval.set_metrics()
    eval.gen_plots()


if __name__ == "__main__":
    main()
