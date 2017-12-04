""" Simulate an ensemble of colliding two-state atoms.
"""

import pygame
import numpy as np

red = (255, 0, 0)
blue = (0, 0, 255)
black = (0, 0, 0)


class Atom:
    """
    A class to represent an atom in pygame.

    Attributes
    ----------
    colour : (int, int, int)
        RGB tuple representing colour of circle to display
    E_kin : float
        Kinetic energy of the atom
    E_ex : float
        Energy stored in the internal state of the atom

    Methods
    -------
    update(delta_t)
        Evolve the atom forward in time by delta_t, and draw it using pygame.
    f_de(E_cm,mu)
        Calculates the de-excitation cross section given the center of mass
        energy, E_cm, and reduced mass, mu.
    """

    def __init__(self, display, size, pos0=None, vel0=None, exc0=0, chi=1e-3,
                 f=0.4, mass=1, g0=1, g1=3, rad=10):
        """
        Constructor for an atom object in pygame.

        Parameters
        ----------
        display : pygame.Surface
            pygame surface on which to display atom
        size : (int, int)
            Size of area within which to constrain the atom

        Keyword Arguments
        -----------------
        pos0 : (int, int)
            Initial position of the atom in units of pixels. If None, the atom
            is placed randomly within the allowed area. (default: None)
        vel0 : (float, float)
            Initial velocity of the atom in units of pixels/ms. If None, the
            velocity components are drawn uniformly between -0.1 and 0.1.
            (default: None)
        exc0 : float
            Initial excitation state. The excitation state is assigned to be
            1 with probability exc0 and 0 otherwise. (default: 0.)
        chi : float
            Energy gap between internal atomic states. (default: 1e-3)
        f : float
            Atomic excitation cross section in units of geometric cross
            section. (default: 0.4)
        mass : float
            Atomic mass (default: 1.)
        g0 : float
            Ground state degeneracy (default: 1.)
        g1 : float
            Excited state degeneracy (default: 3.)
        rad : float
            Radius of atom in units of pixels, used for both displaying and
            calculating collision cross section. (default: 10.)
        """

        self.display = display
        self.disp_size = size
        if pos0 is None:
            self.pos = np.random.rand(2)*self.disp_size
        else:
            self.pos = np.array(pos0).astype(float)
        if vel0 is None:
            self.vel = np.random.uniform(-0.1, 0.1, 2)
        else:
            self.vel = np.array(vel0).astype(float)
        self.exc = int(np.random.rand() < exc0)
        self.chi = float(chi)
        self.f_ex = float(f)
        self.mass = float(mass)
        self.rad = int(rad)
        self.g0 = float(g0)
        self.g1 = float(g1)

    def _draw(self):
        """
        Draw the atom on the pygame surface.
        """

        pygame.draw.circle(self.display, self.colour,
                           self.pos.astype(int),
                           self.rad)
        pygame.draw.circle(self.display, black,
                           self.pos.astype(int),
                           self.rad, 2)

    def _update_position(self, delta_t):
        """
        Evolve the atom forward in time by delta_t based on its current
        velocity.

        Parameters
        ----------
        delta_t : float
            Evolution timestep in ms.
        """

        self.pos += self.vel*delta_t

    def _wall_bounce(self):
        """
        Flip the atoms velocity if it collides with the boundary.
        """

        if (self.pos[0] < self.rad):
            self.vel[0] = abs(self.vel[0])
        elif (self.pos[0] > self.disp_size[0]-self.rad):
            self.vel[0] = -abs(self.vel[0])
        if (self.pos[1] < self.rad):
            self.vel[1] = abs(self.vel[1])
        elif (self.pos[1] > self.disp_size[1]-self.rad):
            self.vel[1] = -abs(self.vel[1])

    def update(self, delta_t):
        """
        Evolve the atom forward in time by delta_t, and draw it using pygame.
        """
        self._update_position(delta_t)
        self._wall_bounce()
        self._draw()

    def f_de(self, E_cm, mu):
        """
        Calculate the de-excitation cross section.

        Parameters
        ----------
        E_cm : float
            Collision center of mass energy
        mu : float
            Reduced mass

        Returns
        -------
        float
            Atomic de-excitation cross section in units of geometric cross
            section.

        Notes
        -----
        The cross section is different from the expression on astrobaki since
        this simulation is 2 dimensional, rather than 3.
        """

        v = np.sqrt(2*E_cm/mu)
        v_prime = np.sqrt(2*(E_cm+self.chi)/mu)
        return (self.g0/self.g1)*(v_prime/v)*self.f_ex

    @property
    def colour(self):
        if self.exc == 0:
            return blue
        else:
            return red

    @property
    def E_kin(self):
        return 0.5*self.mass*np.linalg.norm(self.vel)**2

    @property
    def E_ex(self):
        return self.exc*self.chi


class Ensemble:
    """

    A class to represent an ensemble of identical atoms in pygame.

    Attributes
    ----------
    atoms: list
        List containing all Atom objects in the ensemble
    E_kin : float
        Total kinetic energy of atoms in the ensemble
    E_ex : float
        Total energy stored in internal atomic states in the ensemble
    E_tot : float
        Total energy in the ensemble
    T_kin : float
        Kinetic temperature of the ensemble
    T_ex : float
        Excitation temperature of the ensemble

    Methods
    -------
    update(delta_t)
        Evolve all atoms forward in time by delta_t and draw them using pygame.

    """

    def __init__(self, display, size, n_atoms=40, exc0=0, chi=1e-3,
                 f=0.4, mass=1, g0=1, g1=3, rad=10):
        """
        Constructor for an object representing an ensemble of atoms in pygame.

        Parameters
        ----------
        display : pygame.Surface
            pygame surface on which to display atom
        size : (int, int)
            Size of area within which to constrain the atom

        Keyword Arguments
        -----------------
        n_atoms: int
            Number of atoms in the ensemble. (default: 40)
        exc0 : float
            Initial excitation state. The excitation state is assigned to be
            1 with probability exc0 and 0 otherwise.
        chi : float
            Energy gap between internal atomic states. (default: 1e-3)
        f : float
            Atomic excitation cross section in units of geometric cross
            section. (default: 0.4)
        mass : float
            Atomic mass (default: 1.)
        g0 : float
            Ground state degeneracy (default: 1.)
        g1 : float
            Excited state degeneracy (default: 3.)
        rad : float
            Radius of atoms in units of pixels, used for both displaying and
            calculating collision cross section. (default: 10.)
        """

        self.display = display
        self.disp_size = np.array(size).astype(int)
        self.n_atoms = int(n_atoms)
        self.exc0 = float(exc0)
        self.chi = float(chi)
        self.f = float(f)
        self.mass = float(mass)
        self.g0 = float(g0)
        self.g1 = float(g1)
        self.rad = int(rad)

        self.atoms = [Atom(self.display, self.disp_size, exc0=self.exc0,
                           chi=self.chi, f=self.f, mass=self.f,
                           g0=self.g0, g1=self.g1, rad=self.rad)
                      for i in range(self.n_atoms)]
        self.overlap = []

    def _update_positions(self, delta_t):
        """
        Evolve the ensemble forward in time by delta_t.

        Parameters
        ----------
        delta_t : float
            Evolution timestep in ms.
        """

        for atom in self.atoms:
            atom.update(delta_t)

    def _overlapping(self, atom1, atom2):
        """
        Check if two atoms are overlapping.

        Parameters
        ----------
        atom1 : Atom
            First atom
        atom2 : Atom
            Second atom

        Return
        ------
        bool
            Returns True if atoms are overlapping and False otherwise.
        """

        if np.linalg.norm(atom1.pos-atom2.pos) < (atom1.rad+atom2.rad):
            return True
        else:
            return False

    def _get_collisions(self):
        """
        Get all pairs of atoms in the ensemble that collide.

        Return
        ------
        list
            List of tuples containing pairs of indices corresponding to all
            colliding atoms.

        Notes
        -----
        Two atoms are defined to be colliding if they are overlapping in one
        timestep, when they were not in the previous timestep, or if they have
        collided with another atom since last colliding with one another.
        """

        collisions = []
        for i in range(self.n_atoms):
            for j in range(i+1, self.n_atoms):
                if self._overlapping(self.atoms[i], self.atoms[j]):
                    if not((i, j) in self.overlap):
                        collisions.append((i, j))
                else:
                    try:
                        self.overlap.remove((i, j))
                    except ValueError:
                        pass

        for i, j in collisions:
            for entry in self.overlap:
                if i in entry or j in entry:
                    self.overlap.remove(entry)

        self.overlap += collisions
        return collisions

    def _update_excitation(self, collision_indices):
        """
        Change the excitation states and velocities for colliding atoms.

        Parameters
        ----------
        collision_indices : (int, int)
            Tuple containing the indices of the colliding atoms.
        """

        atom1 = self.atoms[collision_indices[0]]
        atom2 = self.atoms[collision_indices[1]]

        m_tot = atom1.mass+atom2.mass
        mu = atom1.mass*atom2.mass/(m_tot)
        v_cm = (atom1.mass*atom1.vel+atom2.mass*atom2.vel)/m_tot

        v_1_cm = atom1.vel-v_cm
        v_2_cm = atom2.vel-v_cm

        v_1_cm_hat = v_1_cm/np.linalg.norm(v_1_cm)
        v_2_cm_hat = v_2_cm/np.linalg.norm(v_2_cm)

        E_cm = 0.5*atom1.mass*np.linalg.norm(v_1_cm)**2\
            + 0.5*atom2.mass*np.linalg.norm(v_2_cm)**2

        if atom1.exc == 0 and atom2.exc == 0:
            if E_cm > (atom1.chi + atom2.chi):
                if np.random.rand() < atom1.f_ex:
                    E_cm -= atom1.chi
                    atom1.exc = 1

                if np.random.rand() < atom1.f_ex:
                    E_cm -= atom2.chi
                    atom2.exc = 1

            elif max(atom1.chi, atom2.chi) < E_cm < (atom1.chi + atom2.chi):
                if np.random.rand() < 0.5:
                    if np.random.rand() < atom1.f_ex:
                        E_cm -= atom1.chi
                        atom1.exc = 1

                else:
                    if np.random.rand() < atom1.f_ex:
                        E_cm -= atom2.chi
                        atom2.exc = 1

            elif min(atom1.chi, atom2.chi) < E_cm < max(atom1.chi, atom2.chi):
                if atom1.chi < atom2.chi:
                    if np.random.rand() < atom1.f_ex:
                        E_cm -= atom1.chi
                        atom1.exc = 1

                else:
                    if np.random.rand() < atom2.f_ex:
                        E_cm -= atom2.chi
                        atom2.exc = 1

        elif atom1.exc == 0 and atom2.exc == 1:
            if E_cm > atom1.chi and np.random.rand() < atom1.f_ex:
                E_cm -= atom1.chi
                atom1.exc = 1

            if np.random.rand() < atom2.f_de(E_cm, mu):
                E_cm += atom2.chi
                atom2.exc = 0

        elif atom1.exc == 1 and atom2.exc == 0:
            if E_cm > atom2.chi and np.random.rand() < atom2.f_ex:
                E_cm -= atom2.chi
                atom2.exc = 1

            if np.random.rand() < atom1.f_de(E_cm, mu):
                E_cm += atom1.chi
                atom1.exc = 0

        elif atom1.exc == 1 and atom2.exc == 1:
            if np.random.rand() < atom1.f_de(E_cm, mu):
                E_cm += atom1.chi
                atom1.exc = 0

            if np.random.rand() < atom2.f_de(E_cm, mu):
                E_cm += atom2.chi
                atom2.exc = 0

        p_cm = np.sqrt(2*(E_cm)*mu)
        v_2_cm = p_cm*v_2_cm_hat/atom2.mass
        v_1_cm = p_cm*v_1_cm_hat/atom1.mass

        self.atoms[collision_indices[0]].vel = v_cm+v_1_cm
        self.atoms[collision_indices[1]].vel = v_cm+v_2_cm

    def _collide(self):
        """
        Update excitation states and velocities of colliding atoms.
        """

        collisions = self._get_collisions()
        for collision in collisions:
            self._update_excitation(collision)
            atom1 = self.atoms[collision[0]]
            atom2 = self.atoms[collision[1]]

            r = atom1.pos-atom2.pos
            r_mag = np.linalg.norm(r)
            r_hat = r/r_mag

            v_1_r = np.dot(atom1.vel, r_hat)
            v_2_r = np.dot(atom2.vel, r_hat)

            v_1_r_f = (atom1.mass-atom2.mass)*v_1_r/(atom1.mass + atom2.mass)\
                + 2*atom2.mass*v_2_r/(atom1.mass + atom2.mass)
            v_2_r_f = (atom2.mass-atom1.mass)*v_2_r/(atom1.mass + atom2.mass)\
                + 2*atom1.mass*v_1_r/(atom1.mass + atom2.mass)

            delta_v_1 = (v_1_r_f - v_1_r)*r_hat
            delta_v_2 = (v_2_r_f - v_2_r)*r_hat

            self.atoms[collision[0]].vel += delta_v_1
            self.atoms[collision[1]].vel += delta_v_2

    def update(self, delta_t):
        """
        Evolve the ensemble forward in time by delta_t, and draw it.
        """

        self.display.fill((255, 255, 255))
        for atom in self.atoms:
            atom.update(delta_t)
        self._collide()

    @property
    def E_kin(self):
        return np.sum([i_atom.E_kin for i_atom in self.atoms])

    @property
    def E_ex(self):
        return np.sum([i_atom.E_ex for i_atom in self.atoms])

    @property
    def E_tot(self):
        return self.E_kin+self.E_ex

    @property
    def T_kin(self):
        return 1000*(self.E_kin/self.n_atoms)

    @property
    def T_ex(self):
        n1 = np.sum([i_atom.exc for i_atom in self.atoms])
        n0 = self.n_atoms - n1
        if n1 == 0:
            return 0.
        else:
            return 1000*(self.chi/np.log(float(n0)*self.g1/float(n1)/self.g0))


class Display:
    """

    A class to represent a pygame surface displaying an ensemble of atoms.

    """

    def __init__(self, n_atoms=40, fps=30, disp_width=600, disp_height=600,
                 plot_width=600, dpi=100, plot_window=200, exc0=0, chi=1e-3,
                 f=0.4, mass=1, g0=1, g1=3, rad=10):
        """
        Constructor for a screen to display an ensemble of atoms.

        Keyword Arguments
        -----------------
        n_atoms : int
            Number of atoms in the ensemble. (default: 40)
        fps : int
            Frames per second with which to animate the ensemble. (default: 30)
        disp_width : int
            Width of the animation area on the pygame surface. (default: 600)
        disp_height : int
            Height of the animation area on the pygame surface. (default: 600)
        plot_width : int
            Width of the temperature plot accompanying the animation.
            (default: 600)
        dpi : int
            Pixels per inch in the pygame surface. (default: 100)
        plot_window : int
            Number of timesteps to display in the temperature plot.
            (default: 200)
        exc0 : float
            Initial excitation state. The excitation state is assigned to be
            1 with probability exc0 and 0 otherwise. (default: 0.)
        chi : float
            Energy gap between internal atomic states. (default: 1e-3)
        f : float
            Atomic excitation cross section in units of geometric cross
            section. (default: 0.4)
        mass : float
            Atomic mass. (default: 1.)
        g0 : float
            Ground state degeneracy. (default: 1.)
        g1 : float
            Excited state degeneracy. (default: 3.)
        rad : float
            Radius of atoms in units of pixels, used for both displaying and
            calculating collision cross section. (default: 10.)
        """

        self.n_atoms = int(n_atoms)
        self.fps = int(fps)
        self.disp_width = int(disp_width)
        self.disp_height = int(disp_height)
        self.plot_width = int(plot_width)
        self.dpi = int(dpi)
        self.plot_window = int(plot_window)

        self.exc0 = float(exc0)
        self.chi = float(chi)
        self.f = float(f)
        self.mass = float(mass)
        self.g0 = float(g0)
        self.g1 = float(g1)
        self.rad = int(rad)

        self.tot_width = self.disp_width+self.plot_width
        self.width_inches = self.plot_width/self.dpi
        self.height_inches = self.disp_height/self.dpi

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.backends.backend_agg as agg
        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=[self.width_inches, self.height_inches],
                              dpi=self.dpi)
        self.ax = self.fig.gca()
        self.canvas = agg.FigureCanvasAgg(self.fig)

        self.game_display = pygame.display.set_mode((self.tot_width,
                                                     self.disp_height))

        pygame.init()
        pygame.display.set_caption('Collisional Excitations')

        self._run_sim()

    def _run_sim(self):
        """
        Run the simulation, displaying the corresponding animation.
        """

        self.ensemble = Ensemble(self.game_display,
                                 (self.disp_width, self.disp_height),
                                 n_atoms=self.n_atoms, exc0=self.exc0,
                                 chi=self.chi, f=self.f, mass=self.mass,
                                 g0=self.g0, g1=self.g1, rad=self.rad)
        self.window_open = True
        self.t = range(self.plot_window)
        self.T_ex = np.ones(self.plot_window)*np.nan
        self.T_ex[-1] = self.ensemble.T_ex
        self.T_kin = np.ones(self.plot_window)*np.nan
        self.T_kin[-1] = self.ensemble.T_kin

        self.plot_T_ex = self.ax.plot(self.t, self.T_ex, 'r',
                                      label='Excitation Temperature')
        self.plot_T_kin = self.ax.plot(self.t, self.T_kin, 'b',
                                       label='Kinetic Temperature')
        self.ax.legend(loc='upper left')
        self.ax.set_ylim(0, 2*self.ensemble.T_kin)
        self.ax.set_xlim(0, self.plot_window)
        self.ax.set_xlabel('Time (frames)')
        self.ax.set_ylabel('Temperature (arb. units)')
        self.ax.tick_params(labeltop=False, labelright=True, right=True)

        self.clock = pygame.time.Clock()
        while self.window_open:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window_open = False

            self.clock.tick(self.fps)
            self.ensemble.update(self.clock.get_time())
            self._update_plot()
            pygame.display.update()

    def _update_plot(self):
        """
        Update the temperature plot accompanying the simulation.
        """

        self.T_ex[:-1] = self.T_ex[1:]
        self.T_ex[-1] = self.ensemble.T_ex
        self.plot_T_ex[0].set_ydata(self.T_ex)
        self.T_kin[:-1] = self.T_kin[1:]
        self.T_kin[-1] = self.ensemble.T_kin
        self.plot_T_kin[0].set_ydata(self.T_kin)
        self.canvas.draw()

        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        surf = pygame.image.fromstring(raw_data,
                                       (self.plot_width, self.disp_height),
                                       "RGB")
        self.game_display.blit(surf, (self.disp_width, 0))


if __name__ == '__main__':
    disp = Display()
