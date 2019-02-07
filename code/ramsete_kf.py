#!/usr/bin/env python3

# Runs Ramsete simulation on decoupled model with nonlinear trajectory

# Avoid needing display if plots aren't being shown
import sys

if "--noninteractive" in sys.argv:
    import matplotlib as mpl

    mpl.use("svg")
    import latexutils

import control as cnt
import frccontrol as frccnt
import math
import matplotlib.pyplot as plt
import numpy as np


def drivetrain(motor, num_motors, m, r, rb, J, Gl, Gr):
    """Returns the state-space model for a drivetrain.
    States: [[left velocity], [right velocity]]
    Inputs: [[left voltage], [right voltage]]
    Outputs: [[left velocity], [right velocity]]
    Keyword arguments:
    motor -- instance of DcBrushedMotor
    num_motors -- number of motors driving the mechanism
    m -- mass of robot in kg
    r -- radius of wheels in meters
    rb -- radius of robot in meters
    J -- moment of inertia of the drivetrain in kg-m^2
    Gl -- gear ratio of left side of drivetrain
    Gr -- gear ratio of right side of drivetrain
    Returns:
    StateSpace instance containing continuous model
    """
    motor = frccnt.models.gearbox(motor, num_motors)

    C1 = -Gl ** 2 * motor.Kt / (motor.Kv * motor.R * r ** 2)
    C2 = Gl * motor.Kt / (motor.R * r)
    C3 = -Gr ** 2 * motor.Kt / (motor.Kv * motor.R * r ** 2)
    C4 = Gr * motor.Kt / (motor.R * r)
    # fmt: off
    A = np.array([[(1 / m + rb**2 / J) * C1, (1 / m - rb**2 / J) * C3],
                  [(1 / m - rb**2 / J) * C1, (1 / m + rb**2 / J) * C3]])
    B = np.array([[(1 / m + rb**2 / J) * C2, (1 / m - rb**2 / J) * C4],
                  [(1 / m - rb**2 / J) * C2, (1 / m + rb**2 / J) * C4]])
    C = np.array([[1, 0],
                  [0, 1]])
    D = np.array([[0, 0],
                  [0, 0]])
    # fmt: on

    return cnt.ss(A, B, C, D)


class Drivetrain(frccnt.System):
    def __init__(self, dt):
        """Drivetrain subsystem.
        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Left velocity", "m/s"), ("Right velocity", "m/s")]
        u_labels = [("Left voltage", "V"), ("Right voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        u_min = np.array([[-12.0], [-12.0]])
        u_max = np.array([[12.0], [12.0]])
        self.k = 0
        frccnt.System.__init__(self, np.zeros((2, 1)), u_min, u_max, dt,
                               nonlinear=True)

        self.data = np.genfromtxt("ramsete_kf.csv", delimiter=",")
        self.data_t = self.data[1:, 0].T
        self.data_y = self.data[1:, 1:3].T
        self.data_pose_x = self.data[1:, 3:6].T
        self.data_pose_r = self.data[1:, 6:9].T
        self.data_u = self.data[1:, 9:11].T

    def create_model(self, states):
        # Number of motors per side
        num_motors = 2.0

        # High and low gear ratios of drivetrain
        Ghigh = 72.0 / 12.0

        # Drivetrain mass in kg
        m = 64
        # Radius of wheels in meters
        r = 0.0746125
        # Radius of robot in meters
        self.rb = 0.6096 / 2.0
        # Moment of inertia of the drivetrain in kg-m^2
        J = 4.0

        # Gear ratios of left and right sides of drivetrain respectively
        Gl = Ghigh
        Gr = Ghigh

        if self.k == 0:
            self.dt = 0.00505
        else:
            self.dt = self.data_t[self.k] - self.data_t[self.k - 1]

        return drivetrain(frccnt.models.MOTOR_CIM, num_motors, m, r, self.rb, J, Gl, Gr)

    def design_controller_observer(self):
        q_vel = 0.95

        q = [q_vel, q_vel]
        r = [12.0, 12.0]
        self.design_lqr(q, r)

        qff_vel = 0.01
        self.design_two_state_feedforward([qff_vel, qff_vel], [12, 12])

        q_vel = 1.0
        r_vel = 0.05
        self.design_kalman_filter([q_vel, q_vel], [r_vel, r_vel])

    def update_plant(self):
        self.x = self.sysd.A @ self.x + self.sysd.B @ self.u
        self.y = self.data_y[:, self.k:self.k + 1]

    def update_controller(self, next_r):
        self.u = self.data_u[:, self.k:self.k + 1]
        self.r = next_r
        self.k += 1


def main():
    dt = 0.00505
    drivetrain = Drivetrain(dt)

    t = drivetrain.data_t

    refs = []
    for k in range(len(t)):
        r = drivetrain.data[k:k + 1, 11:13].T
        refs.append(r)

    plt.figure(1)
    x_rec, ref_rec, u_rec, y_rec = drivetrain.generate_time_responses(t, refs)
    subplot_max = drivetrain.sysd.states + drivetrain.sysd.inputs
    for i in range(drivetrain.sysd.states):
        plt.subplot(subplot_max, 1, i + 1)
        plt.ylabel(drivetrain.state_labels[i])
        if i == 0:
            plt.title("Time-domain responses")
        plt.plot(t, drivetrain.extract_row(x_rec, i), label="Estimated state")
        plt.plot(t, drivetrain.extract_row(ref_rec, i), label="Reference")
        plt.plot(t, drivetrain.extract_row(y_rec, i), label="Output")
        plt.legend()
    for i in range(drivetrain.sysd.inputs):
        plt.subplot(subplot_max, 1, drivetrain.sysd.states + i + 1)
        plt.ylabel(drivetrain.u_labels[i])
        plt.plot(t, drivetrain.extract_row(u_rec, i), label="Control effort")
        plt.legend()

    plt.figure(2)
    plt.plot(drivetrain.data_pose_r[0, :], drivetrain.data_pose_r[1, :],
             label="Reference trajectory")
    plt.plot(drivetrain.data_pose_x[0, :], drivetrain.data_pose_x[1, :],
             label="Ramsete controller")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

    # Equalize aspect ratio
    xlim = plt.xlim()
    width = abs(xlim[0]) + abs(xlim[1])
    ylim = plt.ylim()
    height = abs(ylim[0]) + abs(ylim[1])
    if width > height:
        plt.ylim([-width / 2, width / 2])
    else:
        plt.xlim([-height / 2, height / 2])

    if "--noninteractive" in sys.argv:
        latexutils.savefig("ramsete_kf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
