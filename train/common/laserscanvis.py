#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from common.laserscan import LaserScan, SemLaserScan
from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.modules.ioueval import iouEval, biouEval
from PIL import Image
from numpy import savetxt
import imageio
import torch

knn_params = {
    "knn": 5,
    "search": 5,
    "sigma": 1.0,
    "cutoff": 1.0,
}

# Create the mapping between original labels
OG_CLASSES2LABELS = {
  0: 0,     # "unlabeled"
  1: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}


class LaserScanVis:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(
        self, scan, scan_names, label_names, offset=0, semantics=True, instances=False,
        predictions=None,
    ):
        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.offset = offset
        self.semantics = semantics
        self.instances = instances
        # sanity check
        if not self.semantics and self.instances:
            print("Instances are only allowed in when semantics=True")
            raise ValueError
        # Set lables for the predicted sequence
        self.predictions = predictions
        if predictions is not None:
            self.pred_sem_labels = predictions
        # Set postprocessing
        self.post = KNN(knn_params, 19)

        self.reset()
        self.update_scan()

    def reset(self):
        """Reset."""
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys="interactive", show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color="white", parent=self.canvas.scene
        )
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = visuals.Markers()
        self.scan_view.camera = "turntable"
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)
        # add semantics
        if self.semantics:
            print("Using semantics in visualizer")
            self.sem_view = vispy.scene.widgets.ViewBox(
                border_color="white", parent=self.canvas.scene
            )
            self.grid.add_widget(self.sem_view, 0, 1)
            self.sem_vis = visuals.Markers()
            self.sem_view.camera = "turntable"
            self.sem_view.add(self.sem_vis)
            visuals.XYZAxis(parent=self.sem_view.scene)
            # self.sem_view.camera.link(self.scan_view.camera)

        if self.instances:
            print("Using instances in visualizer")
            self.inst_view = vispy.scene.widgets.ViewBox(
                border_color="white", parent=self.canvas.scene
            )
            self.grid.add_widget(self.inst_view, 0, 2)
            self.inst_vis = visuals.Markers()
            self.inst_view.camera = "turntable"
            self.inst_view.add(self.inst_vis)
            visuals.XYZAxis(parent=self.inst_view.scene)
            # self.inst_view.camera.link(self.scan_view.camera)

        # img canvas size
        self.multiplier = 1
        self.canvas_W = 1024
        self.canvas_H = 64
        if self.semantics:
            self.multiplier += 1
        if self.instances:
            self.multiplier += 1

        # new canvas for img
        self.img_canvas = SceneCanvas(
            keys="interactive",
            show=True,
            size=(self.canvas_W, self.canvas_H * self.multiplier),
        )
        # grid
        self.img_grid = self.img_canvas.central_widget.add_grid()
        # interface (n next, b back, q quit, very simple)
        self.img_canvas.events.key_press.connect(self.key_press)
        self.img_canvas.events.draw.connect(self.draw)

        # add a view for the depth
        self.img_view = vispy.scene.widgets.ViewBox(
            border_color="white", parent=self.img_canvas.scene
        )
        self.img_grid.add_widget(self.img_view, 0, 0)
        self.img_vis = visuals.Image(cmap="viridis")
        self.img_view.add(self.img_vis)

        # add semantics
        if self.semantics:
            self.sem_img_view = vispy.scene.widgets.ViewBox(
                border_color="white", parent=self.img_canvas.scene
            )
            self.img_grid.add_widget(self.sem_img_view, 1, 0)
            self.sem_img_vis = visuals.Image(cmap="viridis")
            self.sem_img_view.add(self.sem_img_vis)

        # add instances
        if self.instances:
            self.inst_img_view = vispy.scene.widgets.ViewBox(
                border_color="white", parent=self.img_canvas.scene
            )
            self.img_grid.add_widget(self.inst_img_view, 2, 0)
            self.inst_img_vis = visuals.Image(cmap="viridis")
            self.inst_img_view.add(self.inst_img_vis)

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def update_scan(self):
        # first open data
        self.scan.open_scan(self.scan_names[self.offset])
        if self.semantics:
            if self.predictions is not None:
                # Color using the labels from our predictions
                unproj_argmax = self.post(
                            torch.from_numpy(self.scan.proj_range),
                            torch.from_numpy(self.scan.unproj_range),
                            torch.from_numpy(self.pred_sem_labels[self.offset]).flatten().unsqueeze(0),
                            torch.from_numpy(self.scan.proj_x),
                            torch.from_numpy(self.scan.proj_y),
                        )
                # Uncomment if using without KNN
                # unproj_argmax = self.pred_sem_labels[self.offset]
                # unproj_argmax = unproj_argmax[self.scan.proj_y, self.scan.proj_x]
                pred_np = unproj_argmax.numpy().astype(np.int32)
                self.scan.set_label(pred_np)
            else:
                self.scan.open_label(self.label_names[self.offset])
            self.scan.colorize()

        # then change names
        title = "scan " + str(self.offset) + " of " + str(len(self.scan_names))
        self.canvas.title = title
        self.img_canvas.title = title

        # then do all the point cloud stuff

        # plot scan
        power = 16
        # print()
        # p_x = self.scan.proj_x
        # p_y = self.scan.proj_y
        # print(self.scan.proj_x.shape, self.scan.proj_y.shape)
        # print(self.scan.unproj_range.shape)
        # print(self.scan.proj_mask[p_y, p_x].shape)
        # unproj_argmax = proj_argmax[p_y, p_x]
        # original
        range_data = np.copy(self.scan.unproj_range)
        # print(range_data.max(), range_data.min())
        range_data = range_data ** (1 / power)
        # print(range_data.max(), range_data.min())
        viridis_range = (
            (range_data - range_data.min())
            / (range_data.max() - range_data.min())
            * 255
        ).astype(np.uint8)
        viridis_map = self.get_mpl_colormap("viridis")
        # print(viridis_map)
        viridis_colors = viridis_map[viridis_range]
        self.scan_vis.set_data(
            self.scan.points,
            face_color=viridis_colors[..., ::-1],
            edge_color=viridis_colors[..., ::-1],
            size=1,
        )

        # plot semantics
        if self.semantics:
            self.sem_vis.set_data(
                self.scan.points,
                face_color=self.scan.sem_label_color[..., ::-1],
                edge_color=self.scan.sem_label_color[..., ::-1],
                size=1,
            )

        # plot instances
        if self.instances:
            self.inst_vis.set_data(
                self.scan.points,
                face_color=self.scan.inst_label_color[..., ::-1],
                edge_color=self.scan.inst_label_color[..., ::-1],
                size=1,
            )

        # now do all the range image stuff
        # plot range image

        data = np.copy(self.scan.proj_range)
        data[data > 0] = data[data > 0]**(1 / power)
        data[data < 0] = data[data > 0].min()
        # print(data.max(), data.min())
        data = (data - data[data > 0].min()) / \
            (data.max() - data[data > 0].min())
        self.img_vis.set_data(data)
        self.img_vis.update()

        if self.semantics:
            self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
            self.sem_img_vis.update()

        # if self.instances:
        # self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
        # self.inst_img_vis.update()
        print("Update " + str(self.offset) + " completed")

    def save_scans(self):
        for i in range(len(self.scan_names)):
            self.scan.open_scan(self.scan_names[i])
            if self.semantics:
                self.scan.open_label(self.label_names[i])
                self.scan.colorize()
            # np.savetxt('RV_DATASET\\lidar\\00\\l00_offset_'+str(i)+'.npy',
            #           self.scan.proj_range,
            #           fmt="%.16f")

            # Normalize the image
            data = np.copy(self.scan.proj_range)
            data[data > 0] = data[data > 0]
            data[data < 0] = 0
            data = (data) / data.max() * 65535
            im = Image.fromarray((data).astype(np.uint16))
            imageio.imwrite(
                "RV_DATASET\\lidar_png\\02\\l02_offset_" + str(i) + ".png", im
            )
            np.savetxt(
                "RV_DATASET\\masks\\02\\m02_offset_" + str(i) + ".npy",
                self.scan.proj_sem_label,
                fmt="%i",
            )
            im = Image.fromarray(
                (self.scan.proj_sem_color[..., ::-1] * 255).astype(np.uint8)
            )
            im.save("RV_DATASET\\colored\\02\\c02_offset_" + str(i) + ".png")
            print("Save offset " + str(i) + " completed")

    def evaluate(self) -> None:
        evaluator = iouEval(20, "cpu", ignore=[0])
        for i in range(len(self.scan_names)):
            self.scan.open_scan(self.scan_names[i])
            if self.semantics:
                self.scan.open_label(self.label_names[i])
            x = self.pred_sem_labels[i]
            x = x[self.scan.proj_y, self.scan.proj_x]
            y = np.vectorize(OG_CLASSES2LABELS.get)(self.scan.sem_label).astype(np.uint8)
            #print(x, y)
            evaluator.addBatch(x, y)
            print("Eval " + str(i) + " completed")
        
        m_iou, iou = evaluator.getIoU()
        print("*"*80)
        print("Small iou mock problem")
        print("IoU: ", m_iou)
        print("IoU class: ", iou)
        m_acc = evaluator.getacc()
        print("Acc: ", m_acc)
        print("*"*80)

    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()
        self.img_canvas.events.key_press.block()
        if event.key == "N":
            self.offset += 1
            self.update_scan()
        elif event.key == "B":
            self.offset -= 1
            self.update_scan()
        elif event.key == "Q" or event.key == "Escape":
            self.destroy()

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
            self.img_canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.img_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()
