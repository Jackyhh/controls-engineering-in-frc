import numpy as np
import scipy as sp
from merwe_scaled_sigma_points import *
from unscented_transform import *
from frccontrol.runge_kutta import *

template <int States, int Inputs, int Outputs>
class UnscentedKalmanFilter:
    def __init__(inputs, f, h, state_std_devs, measurement_std_devs):
        self.states = len(state_std_devs)
        self.inputs = inputs
        self.outputs = len(measurement_std_devs)

        self.sigmasF = np.zeros((2 * self.states + 1, self.states))

        self.pts = MerweScaledSigmaPoints(self.states)

        self.f = f
        self.h = h
        self.Q = np.diag(np.square(state_std_devs))
        self.R = np.diag(np.square(measurement_std_devs))
        self.reset()

    def Xhat():
        return self.xHat

    def Xhat(i):
        return self.xHat[i, 0]

    def SetXhat(xHat):
        self.xHat = xHat

    def SetXhat(i, value):
        self.xHat[i, 0] = value

    def Reset():
      self.xHat = np.zeros((self.states, 1))
      self.P = np.zeros((self.states, self.states))

    def Predict(u, dt):
        sigmas = m_pts.sigma_points(self.xHat, self.P)

        for i in range(self.pts.num_sigmas()):
            Eigen::Matrix<double, States, 1> x =
                sigmas.template block<1, States>(i, 0).transpose();
            m_sigmasF.template block<1, States>(i, 0) =
                RungeKutta(m_f, x, u, dt).transpose();

        self.xHat, self.P = UnscentedTransform(self.states, self.states,
                                               self.sigmasF, self.pts.Wm(),
                                               self.pts.Wc(), self.Q)

    def Correct(u, y):
        Correct(u, y, self.h, self.R)

    def Correct(rows, u, y, h, R):
      # Transform sigma points into measurement space
      sigmasH = np.zeros((2 * self.states + 1, rows))
      for i in range((self.pts.num_sigmas())):
          sigmasH[i:i + 1, :] = h(self.sigmasF[i:i + 1, :]).T, u)

      # Mean and covariance of prediction passed through UT
      yHat, Py = UnscentedTransform(self.states, rows, sigmasH, self.pts.Wm(),
                                    self.pts.Wc(), R)

      # Compute cross covariance of the state and the measurements
      Pxy = np.zeros((self.states, rows))
      for i in range(self.pts.num_sigmas()):
          Pxy += self.pts.Wc(i) @
                 (m_sigmasF[i:i+1,:] - m_xHat.T).T @
                 (sigmasH[i:i+1,:] - yHat.T)

      K = Pxy @ np.linalg.inv(Py)

      self.xHat += K @ (y - yHat);
      self.P -= K * Py @ K.transpose();
    }

   private:
    std::function<Vector<States>(const Vector<States>&, const Vector<Inputs>&)>
        m_f;
    std::function<Vector<Outputs>(const Vector<States>&, const Vector<Inputs>&)>
        m_h;
    Eigen::Matrix<double, States, 1> m_xHat;
    Eigen::Matrix<double, States, States> m_P;
    Eigen::Matrix<double, States, States> m_Q;
    Eigen::Matrix<double, Outputs, Outputs> m_R;
    Eigen::Matrix<double, 2 * States + 1, States> m_sigmasF;

    MerweScaledSigmaPoints<States> m_pts;
