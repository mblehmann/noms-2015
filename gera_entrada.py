#!/usr/bin/python

import sys

class Parameters:

  def __init__(self, seed):
    self.rng_seeds = [i for i in range(100, seed*100+1, 100)]

    # network topology (routers only)
    self.graph_file = "wlan_topology.gml"
    # time of the simulation
    self.simulation_time = 240
    self.number_users = 750
    self.locations_range = (2, 9)
    self.user_speed_range = (0.2, 1)
    self.session_length = (0.38, 0.18)
    self.capped_session = 80
    self.consumer_locality = 0.75
    ### STRATEGY
    self.vicinity_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    self.max_replicases = [1, 2, 3, 4, 5, 6]
    self.placement_policies = [0, 1]
    ### BASIC
    self.cache_size = 3
    self.cache_timeout = 3
    self.number_objects = [1, 250]
    self.objects_popularity = 0.44

class Simulation:

  def __init__(self, parameters):
    self.parameters = parameters

  def generate_basic(self):
    for vicinity_size in self.parameters.vicinity_sizes:
      for max_replicas in self.parameters.max_replicases:
        for placement_policy in self.parameters.placement_policies:
          for rng_seed in self.parameters.rng_seeds:
            output_file = "basic-v=" + str(vicinity_size) + "-r=" + str(max_replicas) + "-p=" + str(placement_policy) + "-s=" + str(rng_seed) + ".in"
            output = open(output_file, "w")
            
            output.write("graph_file" + "\t" + str(self.parameters.graph_file) + "\n")
            output.write("simulation_time" + "\t" + str(self.parameters.simulation_time) + "\n")
            output.write("rng_seed" + "\t" + str(rng_seed) + "\n")
            output.write("number_users" + "\t" + str(self.parameters.number_users) + "\n")
            output.write("locations_range" + "\t" + str(self.parameters.locations_range[0]) + "\t" + str(self.parameters.locations_range[1]) + "\n")
            output.write("user_speed_range" + "\t" + str(self.parameters.user_speed_range[0]) + "\t" + str(self.parameters.user_speed_range[1]) + "\n")
            output.write("session_length" + "\t" + str(self.parameters.session_length[0]) + "\t" + str(self.parameters.session_length[1]) + "\n")
            output.write("consumer_locality" + "\t" + str(self.parameters.consumer_locality) + "\n")
            output.write("vicinity_size" + "\t" + str(vicinity_size) + "\n")
            output.write("max_replicas" + "\t" + str(max_replicas) + "\n")
            output.write("placement_policy" + "\t" + str(placement_policy) + "\n")
            output.write("number_objects" + "\t" + str(self.parameters.number_objects[0]) + "\n")
            output.write("objects_popularity" + "\t" + str(self.parameters.objects_popularity) + "\n")
            output.write("capped_session" + "\t" + str(self.parameters.capped_session) + "\n")
            output.write("cache_size" + "\t" + str(self.parameters.cache_size) + "\n")
            output.write("cache_timeout" + "\t" + str(self.parameters.cache_timeout) + "\n")

            output.close()

  def generate_multiple(self):
    for vicinity_size in [1, 2]:
      for max_replicas in [1, 2, 3, 4, 5]:
        for rng_seed in self.parameters.rng_seeds:
          placement_policy = 0
          output_file = "multiple-v=" + str(vicinity_size) + "-r=" + str(max_replicas) + "-p=" + str(placement_policy) + "-s=" + str(rng_seed) + ".in"
          output = open(output_file, "w")
            
          output.write("graph_file" + "\t" + str(self.parameters.graph_file) + "\n")
          output.write("simulation_time" + "\t" + str(self.parameters.simulation_time) + "\n")
          output.write("rng_seed" + "\t" + str(rng_seed) + "\n")
          output.write("number_users" + "\t" + str(self.parameters.number_users) + "\n")
          output.write("locations_range" + "\t" + str(self.parameters.locations_range[0]) + "\t" + str(self.parameters.locations_range[1]) + "\n")
          output.write("user_speed_range" + "\t" + str(self.parameters.user_speed_range[0]) + "\t" + str(self.parameters.user_speed_range[1]) + "\n")
          output.write("session_length" + "\t" + str(self.parameters.session_length[0]) + "\t" + str(self.parameters.session_length[1]) + "\n")
          output.write("consumer_locality" + "\t" + str(self.parameters.consumer_locality) + "\n")
          output.write("vicinity_size" + "\t" + str(vicinity_size) + "\n")
          output.write("max_replicas" + "\t" + str(max_replicas) + "\n")
          output.write("placement_policy" + "\t" + str(placement_policy) + "\n")
          output.write("number_objects" + "\t" + str(self.parameters.number_objects[1]) + "\n")
          output.write("objects_popularity" + "\t" + str(self.parameters.objects_popularity) + "\n")
          output.write("capped_session" + "\t" + str(self.parameters.capped_session) + "\n")
          output.write("cache_size" + "\t" + str(self.parameters.cache_size) + "\n")
          output.write("cache_timeout" + "\t" + str(self.parameters.cache_timeout) + "\n")

          output.close()

seeds = int(sys.argv[1])

parameters = Parameters(seeds)

simulation = Simulation(parameters)
simulation.generate_basic()
simulation.generate_multiple()



