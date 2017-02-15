class Settings:

	game = 'BreakoutDeterministic-v0'

	display = False

	agents = 32

	predictors = 2

	trainers = 2

	device = '/cpu:0'

	dynamic_settings = True
	dynamic_settings_step_wait = 20
	dynamic_settings_init_wait = 10

	gamma = 0.99

	tmax = 5

	max_queue_size = 100
	prediction_batch_size = 128

	global_max_steps  = 100000000

	learning_rate = 0.0007

	rms_decay = 0.99
	rms_momentu = 0.0
	rms_epsilon = 0.1

	gradient_clip_norm = 40.0

	training_batch_size = 0

	experiment_name = 'test'

	