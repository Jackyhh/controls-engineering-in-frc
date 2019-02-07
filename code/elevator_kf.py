#!/usr/bin/env python3

# Runs Ramsete simulation on decoupled model with nonlinear trajectory

# Avoid needing display if plots aren't being shown
import sys

if "--noninteractive" in sys.argv:
    import matplotlib as mpl

    mpl.use("svg")
    import latexutils

import frccontrol as frccnt
import matplotlib.pyplot as plt
import numpy as np


class Elevator(frccnt.System):
    def __init__(self, dt):
        """Elevator subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Position", "m"), ("Velocity", "m/s")]
        u_labels = [("Voltage", "V")]
        self.set_plot_labels(state_labels, u_labels)

        self.k = 0
        frccnt.System.__init__(
            self, np.zeros((2, 1)), np.array([[-12.0]]), np.array([[12.0]]), dt,
            nonlinear=True
        )

        # TODO: update for new CSV
        self.data = np.genfromtxt("elevator_kf.csv", delimiter=",")
        self.data_t = self.data[1:, 0].T
        self.data_y = self.data[1:, 1:2].T
        self.data_u = self.data[1:, 5:6].T

    def create_model(self, states):
        # Number of motors
        num_motors = 2.0
        # Elevator carriage mass in kg
        m = 6.803886
        # Radius of pulley in meters
        r = 0.02762679089
        # Gear ratio
        G = 42.0 / 12.0 * 40.0 / 14.0

        return frccnt.models.elevator(frccnt.models.MOTOR_CIM, num_motors, m, r, G)

    def design_controller_observer(self):
        q = [0.02, 0.4]
        r = [12.0]
        self.design_lqr(q, r)
        self.design_two_state_feedforward(q, r)

        q_pos = 0.05
        q_vel = 1.0
        r_pos = 0.0001
        self.design_kalman_filter([q_pos, q_vel], [r_pos])

    def update_plant(self):
        # TODO: update for new CSV
        self.x = self.sysd.A @ self.x + self.sysd.B @ self.u
        self.y = self.data_y[:, self.k:self.k + 1]

    def update_controller(self, next_r):
        # Number of motors
        num_motors = 2.0
        # Elevator carriage mass in kg
        m = 6.803886
        # Radius of pulley in meters
        r = 0.02762679089
        # Gear ratio
        G = 42.0 / 12.0 * 40.0 / 14.0

        motor = frccnt.models.gearbox(frccnt.models.MOTOR_CIM, num_motors)
        u_grav = m**2 * 9.806 * motor.R * r / (G * motor.Kt)

        # TODO: update for new CSV
        self.u = self.data_u[:, self.k:self.k + 1] + u_grav
        self.r = next_r
        self.k += 1


def main():
    dt = 0.00505
    elevator = Elevator(dt)

    t = elevator.data_t

    refs = []
    for k in range(len(t)):
        # TODO: update for new CSV
        r = np.array([[elevator.data[k:k + 1, 4][0]], [0]])
        refs.append(r)

    plt.figure(1)
    x_rec, ref_rec, u_rec, y_rec = elevator.generate_time_responses(t, refs)
    subplot_max = elevator.sysd.states + elevator.sysd.inputs
    for i in range(elevator.sysd.states):
        plt.subplot(subplot_max, 1, i + 1)
        plt.ylabel(elevator.state_labels[i])
        if i == 0:
            plt.title("Time-domain responses")
            plt.plot(t, y_rec[0, :], label="Output")
            plt.legend()
        plt.plot(t, elevator.extract_row(x_rec, i), label="Estimated state")
        plt.plot(t, elevator.extract_row(ref_rec, i), label="Reference")
        plt.legend()
    if "--noninteractive" in sys.argv:
        latexutils.savefig("ramsete_kf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
