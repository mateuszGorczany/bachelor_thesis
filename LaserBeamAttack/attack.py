# %%
def attack(t_max):
	# randomly pick
	model = []
	theta = []
	for t in range(1, t_max):
		s = 0.1
		q = np.array([1,2])
		q = s * q
		theta_prim = theta + q
		prediction_prim = model.evaluate(xl_theta_prim)
		if prediction_prim >= prediction:
			theata = theta_prim
			conf_prim = conf
			break
		if argmax(f(xl)) is not argmax(f(x)):
			return theta

# %%
def k_attack(t_max, k):
	min_theta = 0
	for i in range(1, k):
		theta = attack(t_max)
		if theta < min_theta:
			min_theta = theta
