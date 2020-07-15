'''Written by Josh Surya, for the purpose of solving photorefractive dynamics'''

from twm import *
from scipy.integrate import solve_ivp as solve
from scipy.integrate import BDF
from tqdm import tqdm
from tqdm import trange
from glob import glob
import math

class photorefraction(twm):

	def update(self,hold_s=10,stop_time=2):
		self.thermal_coeff = -1j*self.Ga*self.Ka*5/100

		# Photorefractive effect constants
		self.hold_s = 10 # how many seconds to hold for the last wavelength step
		self.stop_time = 2 # seconds stop at each wavelength
		self.g_e = 1j*1e6
		self.gamma_e = 0.001
		self.Ke = 0.017e0 # 0.017e0 would equal to self.pr_coeff
		self.pr_coeff = 1j*self.Ga*self.Ka*1
		self.Na = 1.0e6 # number of intracavity photons that saturates the photorefractive effect
		self.data = {}
		self.data['intracavity'] = []
		self.data['transmission'] = []
		self.data['ss_intracavity'] = []
		self.data['ss_transmission'] = []
		self.data['Esp'] = []
		self.time_passed = 0
		self.data['time'] = []
		self.data['b'] = []
		self.test = 0
		self.gamma_E = 1 # photorefractive relaxation rate
		self.Esp = 0
		self.set_new = 0
		self.check_roots = {}
		self.check_roots[0] = []
		self.check_roots[1] = []
		pass

	def dEspdt2(self,t,w,w_p,k_vary,eps,Esp):
		self.b = self.thermal_bmode_ss_intracavity(w,w_p,k_vary,eps,self.b,Esp=Esp)
		if self.b <= self.Na/2:
			val = -self.gamma_e * Esp + self.Ke * (1 - self.b / self.Na) * self.b 
		elif self.b == 0:
			val = 0
		else:
			val = -self.gamma_e * Esp + self.Ke * self.Na / 4
		return val

	def dbdt(self,t,b,w,w_p,k_vary,E_sp,eps):
		dbdt = (-1j * (w - w_p) - k_vary) * b - self.thermal_coeff*np.abs(b)**2 * b - self.g_e * E_sp * b + eps
		return dbdt

	def func2(self,t,y,w,w_p,k_vary,eps):
		Esp = y[0]
		dEspdt = self.dEspdt2(t,w,w_p,k_vary,eps,Esp)
		return dEspdt

	def thermal_bmode_ss_intracavity(self,w_b,w_fb,kb,epsilon_pb,ss_b_prev,Esp=None,count=0):
		if epsilon_pb == 0:
			val = 0
		else:
			Xa = -1 * (1j * (w_b - w_fb) + kb)
			Kt = self.thermal_coeff
			zeroth = -1 * np.abs(-1 * epsilon_pb) ** 2
			first = np.abs(Xa)**2+np.abs(self.g_e)**2*np.abs(Esp)**2-2*np.real(self.g_e*Esp*np.conj(Xa))
			second = -2*np.real(Kt*np.conj(Xa))+2*np.real(self.g_e*Esp*np.conj(Kt))
			third = np.abs(Kt)**2
			try:
				coeffs = [third[0], second[0], first[0], zeroth[0]] # there are times when initial w_a/b/c are inputted as an array
			except:
				coeffs = [third, second, first, zeroth]
			roots = np.roots(coeffs)
			ss_b = np.real(roots[np.isreal(roots)])[np.where(np.real(roots[np.isreal(roots)]) > 0)]
			temp = np.abs(ss_b-ss_b_prev)
			val = ss_b[np.where(temp==np.amin(temp))][0]
		return val

	def main_dynamic_Esp_solve(self,scan_speed=2,filename1=None,filename2=None,save=False,single_scan=False): # scan speed defined by nm per second.
		hold_time = 1/((self.lambda_N/ (np.abs(self.lambda_fb[-1]-self.lambda_fb[0])*1e9))*scan_speed)
		t_span1 = (0,hold_time)
		t_span2 = (1e-8,1e-5)
		Esp0 = 0
		b0 = 0
		init_val = [self.Esp]
		count = 0
		if single_scan == True:
			progress = tqdm(total=self.lambda_N)
			start = time.time()
		
		for i,Qcb in enumerate(self.Qcb):
			for j,Qcc in enumerate(self.Qcc):
				for l,w_fb in enumerate(self.w_fb): # [self.w_fb[56]]
					ss_b_prev = self.b
					sol = solve(lambda t, y: self.func2(t,y,self.w_b,w_fb,self.kb[i],self.epsilon_pb[l]), t_span1, y0=init_val, method='LSODA')
					temp_t = sol.t
					temp_Esp = sol.y[0,:]
					Esp0 = sol.y[0,-1]
					init_val = [Esp0]
					trans = self.thermal_bmode_ss_dynamic_trans(self.w_b,w_fb,self.kb[i],self.k1b[i],self.b,Esp0)
					self.data['intracavity'].append(self.thermal_bmode_ss_intracavity(self.w_b,w_fb,self.kb[i],self.epsilon_pb[l],ss_b_prev,Esp0))
					self.data['transmission'].append(trans)
					self.data['Esp'].append(Esp0)
					if single_scan == True:
						progress.update(1)

		if single_scan == True:
			progress.close()
			end = time.time()
			self.Esp = Esp0
			print('time elapsed = ' + str(end - start) + ' seconds')
			print('hold time = ',hold_time,' seconds')
			print('real time passed = ',hold_time*self.lambda_N,' seconds')
			print('Esp = ',Esp0)
		if save == False:
			self.plot_any(self.data['intracavity'],ylabel='photons')
			self.plot_any(self.data['transmission'],ylabel='transmission')
			self.plot_any(self.data['Esp'],ylabel='E-field Amplitude')
		else:
			self.save_data(self.data,filename1,filename2)

	def thermal_bmode_ss_dynamic_trans(self,w_b,w_fb,kb,k1b,ss_b,Esp=None):
		Xa = -1 * (1j * (w_b - w_fb) + kb)
		Kt = self.thermal_coeff
		Na = self.Na
		if Esp is not None:
			K_E = self.g_e*Esp
		else:
			K_E = self.pr_coeff
		top = 2 * k1b
		bottom = Xa - Kt * ss_b - K_E
		val = np.abs(1+top/bottom)**2
		return val

	def thermal_dynamic_trans(self,w_b,w_fb,kb,k1b,ss_b,Esp=None):
		Xa = -1 * (1j * (w_b - w_fb) + kb)
		Kt = self.thermal_coeff
		Na = self.Na
		if Esp is not None:
			K_E = self.g_e*Esp
		else:
			K_E = self.pr_coeff
		if np.abs(ss_b)**2 <= Na/2:
			top = 2 * k1b
			bottom = Xa - Kt * ss_b - K_E * ss_b * (1 - ss_b / Na)
		else:
			top = 2 * k1b
			bottom = Xa - Kt * ss_b - K_E * Na/4
		val = np.abs(1+top/bottom)**2
		return val

	def thermal_ss_intracavity(self,w_b,w_fb,kb,epsilon_pb,ss_b_prev,Esp=None,count=0):
		Xa = -1 * (1j * (w_b - w_fb) + kb)
		Kt = self.thermal_coeff
		if Esp is not None:
			K_E = self.g_e*Esp
		else:
			K_E = self.pr_coeff
		pdb.set_trace()
		Na = self.Na
		zeroth = -1 * np.abs(-1 * epsilon_pb) ** 2
		if ss_b_prev < Na/2:
			first = np.abs(Xa)**2
			second = -2*(np.real(Kt*np.conj(Xa))+np.real(K_E*np.conj(Xa))) #-1*(Kt*np.conj(Xa)+np.conj(Kt)*Xa)
			third = (2/Na)*(np.real(K_E*np.conj(Xa))+np.real(K_E*np.conj(Kt))+(Na/2)*(np.abs(Kt)**2+np.abs(K_E)**2))
			fourth = (2/Na)*(np.real(K_E*np.conj(Kt))-np.abs(K_E)**2)
			fifth = np.abs(K_E)**2/Na**2
			try:
				coeffs = [fifth[0], fourth[0], third[0], second[0], first[0], zeroth[0]] # there are times when initial w_a/b/c are inputted as an array
			except:
				coeffs = [fifth, fourth, third, second, first, zeroth]
			roots = np.roots(coeffs)
			ss_b = np.real(roots[np.isreal(roots)])[np.where(np.real(roots[np.isreal(roots)]) > 0)]
			temp = ss_b-ss_b_prev
			val = ss_b[np.where(temp==np.amin(temp))][0]
			self.last = val.copy()
		else:
			first = np.abs(Xa)**2 - (Na/2)*np.real(Xa*np.conj(K_E)) + np.abs(K_E)**2 * Na**2 / 16
			second = -2*np.real(Kt*np.conj(Xa))+(Na/2)*np.real(Kt*np.conj(K_E))
			third = np.abs(Kt)**2
			try:
				coeffs = [third[0], second[0], first[0], zeroth[0]] # there are times when initial w_a/b/c are inputted as an array
			except:
				coeffs = [third, second, first, zeroth]
			roots = np.roots(coeffs)
			ss_b = np.real(roots[np.isreal(roots)])[np.where(np.real(roots[np.isreal(roots)]) > 0)]
			temp = ss_b-ss_b_prev
			val = ss_b[np.where(temp==np.amin(temp))][0]
		return val

	def thermal_ss_trans(self,w_b,w_fb,kb,k1b,ss_b):
		Xa = -1 * (1j * (w_b - w_fb) + kb)
		Kt = self.thermal_coeff
		Na = self.Na
		K_E = self.pr_coeff
		if ss_b < Na/2:
			top = 2 * k1b
			bottom = Xa - Kt * ss_b - K_E * ss_b * (1 - ss_b / Na)
		else:
			top = 2 * k1b
			bottom = Xa - Kt * ss_b - K_E * Na/4
		val = np.abs(1+top/bottom)**2
		return val
########################################################################################
class laser:
	def __init__(self,wavelength=None,start=None,end=None,scan_speed=None,sample_rate=None,
				total_time=None,Pin=None,modes=None,λ_N=None):
		self.C = 299792458
		self.ħ = 1.054571817e-34
		self.λ = wavelength
		self.ω = 2 * np.pi * self.C / self.λ
		self.start = start
		self.end = end
		self.scan_speed = scan_speed
		self.sample_rate = sample_rate
		self.total_t = total_time
		self.scan_time = self.scan_time(start,end,scan_speed) # time it takes to scan from start to end wavelength
		if λ_N is not None:
			self.λ_N = λ_N
		else:
			self.λ_N = self.lambda_N(self.scan_time,sample_rate)
		self.λ_arr = np.linspace(self.start,self.end,self.λ_N)
		self.ω_arr = 2 * np.pi * self.C / self.λ_arr
		self.κ_arr = []
		self.κ0_arr = []
		self.κ1_arr = []
		for mode in modes:
			if mode.vis:
				self.κ_vis = 2 * self.ω_arr / mode.Ql
				self.κ0_vis = 2 * self.ω_arr / mode.Q0
				self.κ1_vis = 2 * self.ω_arr / mode.Qc
			else:
				self.κ_ir = self.ω_arr / mode.Ql
				self.κ0_ir = self.ω_arr / mode.Q0
				self.κ1_ir = self.ω_arr / mode.Qc
		self.Pin = Pin
		self.hold_time = self.scan_time / self.λ_N
		self.ϵ_p = np.sqrt(2 * modes[0].κ1 * self.Pin / (self.ħ * self.ω_arr))
	
	def lambda_N(self,scan_time,sample_rate):
		return int(scan_time * sample_rate)
		
	def scan_time(self,start,end,scan_speed):
		return np.abs(end-start) / scan_speed
		
class cavity_mode:
	def __init__(self,wavelength=None,Q0=None,Qc=None,Kt=16.7425e10,γ_t=256e3,g=200e3,vis=False,pump=False,aux=False):
		self.C = 299792458
		self.ħ = 1.054571817e-34
		self.λ = wavelength
		self.Q0 = Q0
		if Qc is not list: self.Qc = np.array([Qc]) 
		else: self.Qc = np.array(Qc)
		self.Ql = (self.Q0 ** -1 + self.Qc ** -1) ** -1
		self.ω = 2 * np.pi * self.C / wavelength
		self.κ = self.ω / self.Ql
		self.κ0 = self.ω / self.Q0
		self.κ1 = self.ω / self.Qc
		self.vis = vis
		self.pump = pump
		
		# thermal constants
		self.γ_t = γ_t
		self.Kt = Kt
		self.G = 1e-7
		self.k_th = self.G * self.γ_t
		self.K_T = -1j * self.G * self.Kt * 1*5 / 10000
		self.thermal_coeff = self.K_T
		
		# photorefractive constants
		self.g_e = 1j*1e6
		self.γ_e = 500 # use ~100 for typical triangular shaped curves.  This is used for millisecond response times
		self.Ke = 50
		self.Na = 1.0e6
		self.pr_coeff = -self.K_T*100/5
		
		# SHG constants
		self.g = g
		
		# number of photons in cavity mode
		self.photons = 0

class solve_twm(photorefraction):
	def __init__(self,lasers=None,modes=None):
		self.lasers = lasers
		self.modes = modes
		assert(type(lasers)==list); assert(lasers is not None)
		if len(lasers)>1:
			assert(lasers[0].λ_N==lasers[1].λ_N)
		self.data = {}
		self.data['intracavity'] = np.zeros(shape=(len(modes),lasers[0].λ_N))
		self.data['transmission'] = np.zeros(shape=(len(lasers),lasers[0].λ_N))
		self.data['shg'] = np.zeros(shape=(len(lasers),lasers[0].λ_N))
		self.data['Esp'] = np.zeros(shape=(1,lasers[0].λ_N))
		self.Esp = 0
		self.a = 0
		self.b = 0
		self.shg_soln = np.zeros(shape=(len(lasers),2))

	def reset(self,lasers,modes):
		if len(lasers)>1:
			assert(lasers[0].λ_N==lasers[1].λ_N)
		assert(type(lasers)==list); assert(lasers is not None)
		self.lasers = lasers
		self.modes = modes
		self.data['intracavity'] = np.zeros(shape=(len(modes),lasers[0].λ_N))
		self.data['transmission'] = np.zeros(shape=(len(lasers),lasers[0].λ_N))
		self.data['shg'] = np.zeros(shape=(len(lasers),lasers[0].λ_N))
		self.data['Esp'] = np.zeros(shape=(1,lasers[0].λ_N))
		self.shg_soln = np.zeros(shape=(len(lasers),2))
	
	def dEsp_shg(self,t,lasers,modes,idx,Esp): #t,w,w_p,k_vary,eps,Esp,g_e,γ_e,Ke,Na,Kt
		Na = modes[0].Na; γ_e = modes[0].γ_e; Ke = modes[0].Ke
		self.shg_soln = self.intracavity_shg(lasers,modes,idx,self.shg_soln,Esp=Esp)
		tot = 0
		for i in range(self.shg_soln.shape[0]):
			tot += self.shg_soln[i][0]+2*self.shg_soln[i][1]
		if tot <= Na/2:
			val = -γ_e * Esp + Ke * (1 - tot / Na) * tot 
		elif tot == 0:
			val = 0
		else:
			val = -γ_e * Esp + Ke * Na / 4
		return val
	
	def func_shg(self,t,y,lasers,modes,idx):
		Esp = y[0]
		dEspdt = self.dEsp_shg(t,lasers,modes,idx,Esp)
		return dEspdt
	
	def intracavity_shg(self,lasers,modes,idx,prev_sol,Esp=None):
		''' Here we want to write a function that, given laser inputs and mode inputs and thermal/photorefractive
		parameters, outputs the intracavity steady-state behavior of the pump and SH power, idx is there
		to indicate which laser wavelength we are currently solving for. Here we also make the 
		assumption that every pump laser is far detuned from other modes other than its own.'''
		assert(len(modes)==2*len(lasers))
		assert(prev_sol.shape[0]==len(lasers))
		g_ea,g_eb,γ_ea,γ_eb,Kea,Keb,Naa,Nab,K_Ta,K_Tb,g=[],[],[],[],[],[],[],[],[],[],[]
		# here we want to make it for the general case of N Laser inputers and M modes. Where lasers
		# are at the front of the array and the modes are at the back.
		for jdx,mode in enumerate(modes):
			if mode.pump is True:
				g_ea.append(mode.g_e); γ_ea.append(mode.γ_e); Kea.append(mode.Ke)
				Naa.append(mode.Na); K_Ta.append(mode.K_T); g.append(mode.g)
			else:
				g_eb.append(mode.g_e); γ_eb.append(mode.γ_e); Keb.append(mode.Ke)
				Nab.append(mode.Na); K_Tb.append(mode.K_T)
		sol = np.zeros(shape=(len(lasers),2))
		for k,laser in enumerate(lasers):
			nb = prev_sol[k][1]
			Xa = -1*(1j*(modes[k].ω-laser.ω_arr[idx]))-laser.κ_ir[idx]
			Xb = (-2*K_Tb[k]*(prev_sol[k][0]+prev_sol[k][1]*2)-2*g_eb[k]*Esp-
						1*(1j*(modes[k+len(lasers)].ω-2*laser.ω_arr[idx]))-laser.κ_vis[idx])
			ϵ_tot = laser.ϵ_p[idx]
			if ϵ_tot == 0:
				pass
			else:
				zeroth = -1*np.abs(ϵ_tot)**2
				first = (np.abs(Xa)**2 - 2*np.real(g_ea[k]*Esp*np.conj(Xa)) +
						 np.abs(g_ea[k])**2*np.abs(Esp)**2)#np.abs(Xa)**2
				second = (-2*np.real(np.conj(K_Ta[k])*Xa) + 2*np.real(g_ea[k]*Esp*np.conj(K_Ta[k])) + 
						 4*g[k]**2*np.real(Xa/np.conj(Xb)) - 4*g[k]**2*np.real(np.conj(g_ea[k]*Esp)/Xb))#4*g**2*(np.real(Xa/np.conj(Xb)))
				third = (4*g[k]**4/np.abs(Xb)**2 + np.abs(K_Ta[k])**2 - 
						 4*g[k]**2*np.real(np.conj(K_Ta[k])/Xb))
				try:
					coeffs = [third[0], second[0], first[0], zeroth[0]] # there are times when initial w_a/b/c are inputted as an array
				except:
					coeffs = [third, second, first, zeroth]
				try:
					roots = np.roots(coeffs)
					ss_b = np.real(roots[np.isreal(roots)])[np.where(np.real(roots[np.isreal(roots)]) > 0)]
					temp = np.abs(ss_b-prev_sol[k][0])
					sol[k][0] += (ss_b[np.where(temp==np.amin(temp))][0])#[np.where(temp==np.amin(temp))]
					sol[k][1] += (g[k]**2*sol[k][0]**2/np.abs(Xb)**2)
					prev_sol[k][0] = sol[k][0]; prev_sol[k][1] = sol[k][1]
				except:
					pdb.set_trace()
		return sol
	
	def shg_therm_ss_PR_dynamic_trans(self,lasers,modes,idx,prev_sol,Esp=None):
		val = []
		for k,laser in enumerate(lasers):
			Xa = (-1*(1j*(modes[k].ω-laser.ω_arr[idx]))-laser.κ_ir[idx]-
				 modes[k].K_T*(modes[k].photons+prev_sol[k][1]*2)-modes[k].g_e*Esp)
			Xb = (-2*modes[k+len(lasers)].K_T*(prev_sol[k][0]+prev_sol[k][1]*2)-
				  2*modes[k+len(lasers)].g_e*Esp-
				  1*(1j*(modes[k+len(lasers)].ω-2*laser.ω_arr[idx]))-laser.κ_vis[idx])
			top = 2 * laser.κ1_ir[idx]
			bottom = Xa + 2*modes[k].g**2*modes[k].photons/Xb
			val.append(np.abs(1+top/bottom)**2)
		assert(len(val)==len(lasers))
		return val
	
	def pr_shg_transmission(self,fn1=None,fn2=None,save=False,single_scan=False,time_scan=False,many_plots=False):
		# Assume there is only one laser and one mode for now
		t_span1 = (0,self.lasers[0].hold_time)
		t_span2 = (1e-8,1e-5)
		Esp0 = 0
		b0 = 0
		init_val = [self.Esp]
		count = 0
		if single_scan == True:
			progress = tqdm(total=self.lasers[0].λ_N)
			start = time.time()
		
		for i,Qcb in enumerate(self.modes[0].Qc):
#             for j,Qcc in enumerate(self.Qcc):
			for l,w_fb in enumerate(self.lasers[0].ω_arr):
				ss_b_prev = self.shg_soln
				sol = solve(lambda t, y: self.func_shg(t,y,self.lasers,self.modes,l),
							t_span1, y0=init_val, method='LSODA')
				temp_t = sol.t
				temp_Esp = sol.y[0,:]
				Esp0 = sol.y[0,-1]
				init_val = [Esp0]
				for k,laser in enumerate(self.lasers):
					self.modes[k].photons = self.shg_soln[k][0]
					self.modes[k+len(self.lasers)].photons = self.shg_soln[k][1]
				trans = self.shg_therm_ss_PR_dynamic_trans(self.lasers,self.modes,l,ss_b_prev,Esp0)
				self.store_data(self.lasers,self.modes,l,trans,Esp0)
				if single_scan == True:
					progress.update(1)
				if time_scan == True:
					assert(len(self.lasers[0].ω_arr)==1)
					self.time_arr = sol.t
					self.Esp_arr = sol.y[0,:]
					self.sol = sol
		self.Esp = Esp0
		if single_scan == True:
			progress.close()
			end = time.time()
			print('time elapsed = ' + str(end - start) + ' seconds')
			print('hold time = ',self.lasers[0].hold_time,' seconds')
			print('real time passed = ',self.lasers[0].hold_time*self.lasers[0].λ_N,' seconds')
			print('Esp = ',Esp0)
		if (save == False) and (many_plots == False):
			for k in range(len(self.lasers)):
				self.plot_any(self.data['transmission'][k],ylabel='normalized transmission')
				self.plot_any(self.data['intracavity'][k],ylabel='intracavity photons')
				self.plot_any(self.data['intracavity'][k+len(self.lasers)],ylabel='SH photons')
			self.plot_any(self.data['Esp'][k],ylabel='E-field Amplitude')
		elif (save == False) and (many_plots == True):
			# plot_many should be take an input of two lists, the first is the data, the second is ylabels
			data_list = []
			data_labels = []
			for k in range(len(self.lasers)):
				data_list.append(self.data['transmission'][k])
				data_labels.append('normalized transmission')
				data_list.append(self.data['intracavity'][k])
				data_labels.append('intracavity photons')
				data_list.append(self.data['intracavity'][k+len(self.lasers)])
				data_labels.append('SH photons')
			data_list.append(self.data['Esp'][k])
			data_labels.append('E-field Amplitude')
			self.plot_many(data_list,data_labels)
		else:
			self.save_data(self.data,fn1,fn2)
			
	def store_data(self,lasers,modes,idx,trans,Esp):
		for k,laser in enumerate(lasers):
			self.data['intracavity'][k][idx] += modes[k].photons # IR mode
			self.data['intracavity'][k+len(lasers)][idx] += modes[k+len(lasers)].photons # SH mode
			self.data['transmission'][k][idx] += trans[k]
		self.data['Esp'][0][idx] += Esp
	
	def plot_any(self,data, ylabel='set units',sx=8,sy=6,suffix=0):
		fig, ax = plt.subplots(1, 1, figsize=(sx, sy))
		ax.set_xlabel('Pump Wavelength (nm)')
		ax.set_ylabel(ylabel)
		ax.plot(self.lasers[0].λ_arr * 1e9, data)  # self.w_fb/1e12/2/np.pi
		ax.ticklabel_format(useOffset=False)
		plt.show()
	
	def plot_many(self,data,ylabels=None,sx=10,sy=3):
		'''plot many graphs, default x axis is pump wavelength'''
		ynum_plots = math.ceil(len(data)/2)
		fig,ax = plt.subplots(ynum_plots,2,figsize=(sx,ynum_plots*sy))
		idx = 0
		if ylabels:
			assert(len(data)==len(ylabels))
		for yplts in ax:
			for xplt in yplts:
				xplt.plot(self.lasers[0].λ_arr * 1e9,data[idx])
				xplt.set_xlabel('pump λ (nm)')
				if ylabels:
					xplt.set_ylabel(ylabels[idx])
				xplt.ticklabel_format(axis='both',useOffset=False)
				xplt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
				idx += 1

		plt.show()

	def load_datafile(self,filename):
		def f(array_string):
			try:
				val=np.array(eval(array_string))
			except:
				val = np.nan
			return val
		self.df = pd.read_csv(filename,converters = {'intracavity':f,'transmission':f,'shg':f,'Esp':f})
		self.load_data = self.df.to_dict('list')
	