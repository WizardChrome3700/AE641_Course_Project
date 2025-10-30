import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

class Scenario:
    def __init__(self, missile_speed, target_speed, init_LOS, delta, alpha_target, init_Range):
        self.missile_speed = missile_speed
        self.target_speed = target_speed
        self.init_LOS = init_LOS
        self.delta = delta
        self.alpha_target = alpha_target
        self.init_Range = init_Range
    
    def __del__(self):
        pass

    def deviated_pursuit(self, t, state):
        missile_X, missile_Y, target_X, target_Y, LOS_angle = state
        current_range = np.sqrt((missile_X - target_X)**2 + (missile_Y - target_Y)**2)

        # LOS rate (core deviated pursuit equation)
        LOS_rate = (self.target_speed * np.sin(self.alpha_target - LOS_angle) 
                   - self.missile_speed * np.sin(self.delta)) / current_range

        missile_velocity_x = self.missile_speed * np.cos(self.delta + LOS_angle)
        missile_velocity_y = self.missile_speed * np.sin(self.delta + LOS_angle)

        target_velocity_x = self.target_speed * np.cos(self.alpha_target)
        target_velocity_y = self.target_speed * np.sin(self.alpha_target)

        return [missile_velocity_x, missile_velocity_y, target_velocity_x, target_velocity_y, LOS_rate]

    def simulation(self):
        velocity_R_init = self.target_speed * np.cos(self.alpha_target - self.init_LOS) - self.missile_speed * np.cos(self.delta)
        velocity_theta_init = self.target_speed * np.sin(self.alpha_target - self.init_LOS) - self.missile_speed * np.sin(self.delta)
        K = self.missile_speed / self.target_speed
        sin_beta = K * np.sin(self.delta)
        
        if (K > 1) and (abs(sin_beta) < 1):
            t_end_of_sim = (self.init_Range * (velocity_R_init + 
                             2 * self.missile_speed * np.cos(self.delta) - 
                             velocity_theta_init * np.tan(self.delta))) / (self.missile_speed**2 - self.target_speed**2)
            
            print(f"Predicted intercept time: {t_end_of_sim:.2f} seconds")
            print(f"Initial range: {self.init_Range/1000:.2f} km")
            
            missile_x0 = 0
            missile_y0 = 0
            target_x0 = self.init_Range * np.cos(self.init_LOS)
            target_y0 = self.init_Range * np.sin(self.init_LOS)
            LOS0 = self.init_LOS
            initial_state = [missile_x0, missile_y0, target_x0, target_y0, LOS0]
            t_span = [0, t_end_of_sim]
        
            # CORRECTED intercept event
            def intercept_event(t, state):
                missile_x, missile_y, target_x, target_y, LOS_angle = state
                range_current = np.sqrt((missile_x - target_x)**2 + (missile_y - target_y)**2)
                return range_current  # Stop when range < 50 meters
            
            intercept_event.terminal = True
            intercept_event.direction = -1  # Trigger when decreasing through zero
            
            sol = solve_ivp(
                self.deviated_pursuit,
                t_span,
                initial_state,
                method='RK45',
                events=intercept_event,
                rtol=1e-6,
                atol=1e-9
            )
            
            return sol
        else:
            print("Interception not possible in tail-chase.")
            return None
    
    def plot_results(self, sol):
        """Plot the simulation results"""
        if sol is None:
            return
            
        t = sol.t
        missile_x, missile_y, target_x, target_y, LOS_angle = sol.y
        
        # Calculate range history
        range_history = np.sqrt((missile_x - target_x)**2 + (missile_y - target_y)**2)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(missile_x, missile_y, 'r-', label='Missile', linewidth=2)
        plt.plot(target_x, target_y, 'b--', label='Target', linewidth=2)
        plt.plot(missile_x[0], missile_y[0], 'ro', markersize=8, label='Missile Start')
        plt.plot(target_x[0], target_y[0], 'bo', markersize=8, label='Target Start')
        if len(missile_x) > 1:
            plt.plot(missile_x[-1], missile_y[-1], 'rx', markersize=10, label='Missile End')
            plt.plot(target_x[-1], target_y[-1], 'bx', markersize=10, label='Target End')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)
        plt.title('Trajectories')
        plt.axis('equal')
        
        plt.subplot(1, 3, 2)
        plt.plot(t, np.rad2deg(LOS_angle))
        plt.xlabel('Time (s)')
        plt.ylabel('LOS Angle (deg)')
        plt.grid(True)
        plt.title('Line of Sight Angle')
        
        plt.subplot(1, 3, 3)
        plt.plot(t, range_history/1000)
        plt.xlabel('Time (s)')
        plt.ylabel('Range (km)')
        plt.grid(True)
        plt.title('Missile-Target Range')
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./Outputs/Vm{np.format_float_positional(self.missile_speed, 3)}_Vt{np.format_float_positional(self.target_speed, 3)}_thetaInit{np.format_float_positional(self.init_LOS*180/np.pi, 3)}_delta{np.format_float_positional(self.delta*180/np.pi, 3)}_alphaTarget{np.format_float_positional(self.alpha_target*180/np.pi, 3)}_RInit{np.format_float_positional(self.init_Range/1000, 3)}.png')
        
        # Print comprehensive results
        final_range = range_history[-1]
        print(f"\n=== SIMULATION RESULTS ===")
        print(f"Final range: {final_range:.2f} m")
        print(f"Simulation time: {t[-1]:.2f} s")
        print(f"Range reduction: {(self.init_Range - final_range)/1000:.2f} km")
        print(f"Final LOS angle: {np.rad2deg(LOS_angle[-1]):.2f}°")

# Test the simulation
scenario_df = pd.read_csv("Scenarios.csv")
# print(np.array(scenario_df[['Scenario 1']]))
# print(scenario_df.info())
for j in range(len(scenario_df.columns) - 1):
    print(f"\nScenario {j} : ", end='')
    for i in range(scenario_df.shape[0]):
        print(np.transpose(np.array(scenario_df[[' ']]))[0][i], end=" = ")
        if(i == scenario_df.shape[0]-1):
            print(np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][i], end='\n')
        else:
            print(np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][i], end=', ')
    scene = Scenario(missile_speed=np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][0], target_speed=np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][1], 
                    init_LOS=np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][2]*np.pi/180, delta=np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][3]*np.pi/180, 
                    alpha_target=np.transpose(np.array(scenario_df[[f'Scenario {j+1}']]).astype(float))[0][4]*np.pi/180, init_Range=50e3)
    sol = scene.simulation()
    if sol is not None:
        scene.plot_results(sol)
    del scene