#!/usr/bin/python

# IMPORTS
import sys, time, math, bisect, numpy
from graph_tool.all import * # graph library

# Simulator version
# year.month.day.version-author
VERSION = "2015.09.10.1-ML"

begin_time = time.time()

# Some variables

# Event index dictionary
EVENTS = {'START_SESSION': 0, 'START_MOVEMENT': 1, 'CACHE_TIMEOUT': 2, 'CONTENT_CREATION': 3, 'CONTENT_PUSH': 4, 'CONSUMER_REQUEST': 5, 'END_TIME': 6, 'DEBUG': 7}

# Placement Policy Index
RANDOM_DEVICE = 0
BEST_AVAILABLE_DEVICE = 1

# General Error
def print_help():
  print 'Error'

### INPUT
if len(sys.argv) < 2:
  print_help()
  sys.exit(1)

# Generate a Zipf distribution
class ZipfGenerator: 

  def __init__(self, n, alpha): 
    # Calculate Zeta values from 1 to n: 
    tmp = [1. / (math.pow(float(i), alpha)) for i in range(1, n+1)] 
    zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0]) 

    # Store the translation map: 
    self.distMap = [x / zeta[-1] for x in zeta] 

  def next(self): 
    # Take a uniform 0-1 pseudo-random value: 
    u = numpy.random.random()  

    # Translate the Zipf variable: 
    return bisect.bisect(self.distMap, u) - 1

### Event Queue

# This queue implements the priority feature to execute actions in a certain order
class Queue:
  def __init__(self):
    self.queue = {}
    for event in range(len(EVENTS)):
      self.queue[event] = {}

  # Add a new event to the queue with given time
  def add_event(self, time, event):
    event_index = event[0]
    if time not in self.queue[event_index]:
      self.queue[event_index][time] = []
    self.queue[event_index][time].append(event)

  def next_timestep(self):
    timestep = None
    for priority in range(len(EVENTS)):
      if len(self.queue[priority]) > 0:
        if timestep == None:
          timestep = min(self.queue[priority])
        else:
          timestep = min(timestep, min(self.queue[priority]))
    return timestep

  def get_events(self, time):
    events = []
    for priority in range(len(EVENTS)):
      if time in self.queue[priority]:
        events += self.queue[priority][time]
    return events

  def event_in_time(self, time, event_index):
    return time in self.queue[event_index]

  def remove_events(self, time):
    for priority in range(len(EVENTS)):
      if time in self.queue[priority]:
        del self.queue[priority][time]

### Simulator

class Simulator:
  
### VARIABLES

  # Event_queue
  event_queue = Queue()

  # Stores all information about all contents
  contents = {}

  popularity = {}

  factor = 1

  current_session = {}

  # Current timestep of the simulation
  timestep = 0

  # Topology (graph)
  topology = None

  # List of routers
  routers = None

  # Number of routers in the topology
  number_routers = 0

  # List of user devices
  devices = None

  # List of producers (for each content)
  content_producers = {}

  # Distance between routers
  shortest_paths = {}

  eccentricity = {}

  # Retrieval time results
  stats_retrieval_time = {}

  # Hit rate results
  stats_hit_rate = {}

  # Pushing time results
  stats_pushing_time = {}

  # Vicinity cost results
  stats_vicinity_cost = {}

  # Announce time results
  stats_announce_time = {}

  # Announce cost results
  stats_announce_cost = {}

  events_counter = {}

### CONSTRUCTOR

  def __init__(self, input_file):
    input_file_reader = open(input_file, "r")
    for line in input_file_reader:
      line = line.split()
      if line[0] == 'graph_file':
        self.graph_file = line[1]
      elif line[0] == 'simulation_time':
        self.simulation_time = int(line[1])
      elif line[0] == 'rng_seed':
        self.rng_seed = int(line[1])
      elif line[0] == 'number_users':
        self.number_users = int(line[1])
      elif line[0] == 'locations_range':
        self.locations_range = (int(line[1]), int(line[2]))
      elif line[0] == 'user_speed_range':
        self.user_speed_range = (float(line[1]), float(line[2]))
      elif line[0] == 'session_length':
        self.session_length = (float(line[1]), float(line[2]))
      elif line[0] == 'consumer_locality':
        self.consumer_locality = float(line[1])
      elif line[0] == 'vicinity_size':
        self.vicinity_size = int(line[1])
      elif line[0] == 'max_replicas':
        self.max_replicas = int(line[1])
      elif line[0] == 'placement_policy':
        self.placement_policy = int(line[1])
      elif line[0] == 'number_objects':
        self.number_objects = int(line[1])
      elif line[0] == 'objects_popularity':
        self.objects_popularity = float(line[1])
      elif line[0] == 'capped_session':
        self.capped_session = float(line[1])
      elif line[0] == 'cache_size':
        self.cache_size = float(line[1])
      elif line[0] == 'cache_timeout':
        self.cache_timeout = int(line[1])

### SIMULATION SETUP AND INITIALIZATION

  def setup_simulation(self):

    # Initiate the random number generator
    numpy.random.seed(self.rng_seed)

    # Generates the network topology
    self.initialize_topology()
    
    # Set up the user devices position in the network and their movement pattern
    self.initialize_position()

    # Initialize the movement probability
    self.initialize_session()

    self.initialize_popularity()

    # Initialize the contents generated in the simulation
    self.initialize_content()

    # Initialize end of simulation
    self.initialize_end()

  # Initialize Topology and relevant configurations
  def initialize_topology(self):
    self.topology = load_graph(self.graph_file)

    self.setup_vertices_properties()
    self.initialize_routers()
    self.initialize_user_devices()
    self.calculate_routers_path()

  def setup_vertices_properties(self):
    # Setup the vertices properties
    vprop_router = self.topology.new_vertex_property("bool")
    vprop_device = self.topology.new_vertex_property("bool")
    vprop_cache = self.topology.new_vertex_property("object")
    vprop_online = self.topology.new_vertex_property("bool")
    vprop_home = self.topology.new_vertex_property("int")
    vprop_places = self.topology.new_vertex_property("vector<int>")
    vprop_interest = self.topology.new_vertex_property("object")
    vprop_position = self.topology.new_vertex_property("int")
    vprop_created = self.topology.new_vertex_property("vector<int>")
    vprop_announce = self.topology.new_vertex_property("vector<int>")

    self.topology.vertex_properties['router'] = vprop_router
    self.topology.vertex_properties['device'] = vprop_device
    self.topology.vertex_properties['cache'] = vprop_cache
    self.topology.vertex_properties['online'] = vprop_online
    self.topology.vertex_properties['home'] = vprop_home
    self.topology.vertex_properties['places'] = vprop_places
    self.topology.vertex_properties['interest'] = vprop_interest
    self.topology.vertex_properties['position'] = vprop_position
    self.topology.vertex_properties['created'] = vprop_created
    self.topology.vertex_properties['announce'] = vprop_announce

  def initialize_routers(self):
    # Setup routers
    for router in self.topology.vertices():
      self.topology.vertex_properties['router'][router] = True
      self.topology.vertex_properties['device'][router] = False
      self.topology.vertex_properties['cache'][router] = {}
      self.topology.vertex_properties['online'][router] = True
      self.topology.vertex_properties['home'][router] = int(router)
      self.topology.vertex_properties['places'][router] = [int(router)]
      self.topology.vertex_properties['interest'][router] = {}
      self.topology.vertex_properties['position'][router] = int(router)
      self.topology.vertex_properties['created'][router] = []
      self.topology.vertex_properties['announce'][router] = []
    self.number_routers = self.topology.num_vertices()

  def initialize_user_devices(self):
    # Create user devices
    self.topology.add_vertex(self.number_users)

    self.topology.set_vertex_filter(self.topology.vertex_properties['router'], True)

    # Set them up
    for device in self.topology.vertices():
      self.topology.vertex_properties['router'][device] = False
      self.topology.vertex_properties['device'][device] = True
      self.topology.vertex_properties['cache'][device] = {}
      self.topology.vertex_properties['online'][device] = True
      self.topology.vertex_properties['home'][device] = -1
      self.topology.vertex_properties['places'][device] = []
      self.topology.vertex_properties['interest'][device] = {}
      self.topology.vertex_properties['position'][device] = -1
      self.topology.vertex_properties['created'][device] = []
      self.topology.vertex_properties['announce'][device] = []

    self.topology.set_vertex_filter(None)

  def calculate_routers_path(self):
    # Get information to variables
    self.routers = GraphView(self.topology, vfilt=self.topology.vertex_properties['router'])
    self.devices = GraphView(self.topology, vfilt=self.topology.vertex_properties['device'])

    # Distance between routers
    # Triangle matrix
    # Self to self is [self]
    self.shortest_paths = {}
    for source in range(self.number_routers):
      self.eccentricity[source] = 0
      for target in range(source, self.number_routers):
        if source not in self.shortest_paths:
          self.shortest_paths[source] = {}
        if source == target:
          self.shortest_paths[source][source] = [source]
        else:
          vertex_list, edge_list = graph_tool.topology.shortest_path(self.topology, self.topology.vertex(source), self.topology.vertex(target))
          self.shortest_paths[source][target] = [int(vertex) for vertex in vertex_list]

    for source in range(self.number_routers):
      for target in range(self.number_routers):
        self.eccentricity[source] = max(self.eccentricity[source], self.get_distance(source, target))

  # Configure positions in the network
  def initialize_position(self):
    # Set up for each device
    for device in self.devices.vertices():
      self.calculate_locations(device)
      self.setup_initial_position(device)

  # Calculate possible locations of the device
  def calculate_locations(self, device):
    number_locations = numpy.random.randint(self.locations_range[0], self.locations_range[1]+1)
    places = 0
    while places < number_locations:
      location = numpy.random.randint(0, self.number_routers)
      if location not in self.topology.vertex_properties['places'][device]:
        self.topology.vertex_properties['places'][device].append(location)
        places += 1

  # Setup initial position
  def setup_initial_position(self, device):
    starting_location = numpy.random.choice(self.topology.vertex_properties['places'][device])
    self.topology.vertex_properties['home'][device] = starting_location
    self.topology.vertex_properties['position'][device] = starting_location
    self.topology.add_edge(device, self.topology.vertex(starting_location))

  # Setup the movement probability. It takes into consideration the pause probability as described in the paper "Steady-State of the SLAW Mobility Model"
  def initialize_session(self):
    for device in self.devices.vertices():
      self.start_session(device)

  # Sets for each device if it is available (online) or not in this time step
  def start_session(self, device):
    session = self.get_session_length()
    self.current_session[int(device)] = (self.timestep, session)
    self.topology.vertex_properties['online'][device] = True
    self.event_queue.add_event(self.timestep + int(math.ceil(session)), (EVENTS['START_MOVEMENT'], device))
    self.announce(device, available=True)

  # Returns a session length based on a session distribution
  def get_session_length(self):
    return min(self.capped_session, numpy.random.pareto(self.session_length[0]) + self.session_length[1])

  # Setup the content popularity
  def initialize_popularity(self):
    zipf = ZipfGenerator(self.number_objects, self.objects_popularity)
    sample_size = 10000
    for request in range(sample_size):
      content_object = zipf.next()
      if content_object not in self.popularity:
        self.popularity[content_object] = 0
      self.popularity[content_object] += 1

    # Distribution of requests (percentage)
    for content_object in self.popularity:
      self.popularity[content_object] = self.popularity[content_object] / float(sample_size)

    self.factor = max(1, 10 / (max(1, int(math.floor(min(self.popularity.values()) * (self.number_users-1))))))

  # Setup the contents
  def initialize_content(self):
    for content in range(self.number_objects):
      self.schedule_content_object_creation(content)
      self.initialize_statistic_structures(content)

  def schedule_content_object_creation(self, content):
    producer = numpy.random.randint(self.number_routers, self.number_users+self.number_routers)
    time = numpy.random.randint(0, self.simulation_time)
    requests = min(max(1, int(math.floor(self.popularity[content] * (self.number_users-1) * self.factor))), self.number_users-1)

    # Second event of a timestep is content creation
    self.content_producers[content] = producer
    self.event_queue.add_event(time, (EVENTS['CONTENT_CREATION'], (content, producer)))

    # Third is to request data
    # Initialize interests on contents
    self.initialize_interest(content, time, requests)

  def initialize_statistic_structures(self, content):
    self.stats_retrieval_time[content] = {}
    self.stats_hit_rate[content] = {}
    self.stats_pushing_time[content] = []
    self.stats_vicinity_cost[content] = []
    self.stats_announce_time[content] = []
    self.stats_announce_cost[content] = []      

  # Initialize interests upon the contents
  # If they are uniform, the number is pre-determined
  # If they are clustered, a router is selected and the interest is defined according to the distance to that router
  def initialize_interest(self, content, time, requests):
    # The requests are geo-based around the router within a certain distance
    # (according to number of users and locality factor)
    max_distance = 0
    potential_local_consumers = []
    potential_global_consumers = []

    # Initialize the interest
    for device in self.devices.vertices():
      self.topology.vertex_properties['interest'][device][content] = 0
      potential_global_consumers.append(device)

    # There is a router that is the center of requests
    router = numpy.random.randint(0, self.number_routers)

    # While there is not a set of users interested, we expand the radius by 1 router
    while len(potential_local_consumers) < requests * self.consumer_locality:
      potential_local_consumers = []
      max_distance += 1
      for device in self.devices.vertices():
        distance = self.get_distance(router, self.topology.vertex_properties['position'][device])
        if distance < max_distance and int(device) != self.content_producers[content]:
          potential_local_consumers.append(device)

    local_consumers = self.create_requests(content, potential_local_consumers, int(math.floor(requests * self.consumer_locality)), time)
    potential_global_consumers = list(set(potential_global_consumers) - set(local_consumers))
    self.create_requests(content, potential_global_consumers, int(math.floor(requests * (1 - self.consumer_locality))), time)

  def create_requests(self, content, device_set, number_requests, time):
    requests_issued = 0
    consumers = []
    while requests_issued < number_requests:
      device = numpy.random.choice(device_set)
      if self.topology.vertex_properties['interest'][device][content] == 0 and int(device) != self.content_producers[content]:
        request_time = numpy.random.randint(time, self.simulation_time)
        self.schedule_request(device, content, request_time)
        consumers.append(device)
        requests_issued += 1
    return consumers

  # Schedule a content request in the event queue
  def schedule_request(self, device, content, request_time):
    self.topology.vertex_properties['interest'][device][content] = 1
    self.event_queue.add_event(request_time, (EVENTS['CONSUMER_REQUEST'], (int(device), content)))

  def initialize_end(self):
    self.event_queue.add_event(self.simulation_time, (EVENTS['END_TIME'], True))

  # Get the distance from router a to router b
  # The distance is the path minus 1, because we are interested in the number of links and not in the number of routers
  def get_distance(self, source, target):
    origin = min(source, target)
    destination = max(source, target)
    return len(self.shortest_paths[origin][destination]) - 1

  # Get the path from router a to router b
  def get_path(self, source, target):
    origin = min(source, target)
    destination = max(source, target)
    return self.shortest_paths[origin][destination]

  # Checks if the device is home
  def is_device_home(self, device):
    return self.topology.vertex_properties['position'][device] == self.topology.vertex_properties['home'][device]

  # Checks if the device is the producer (owner) of the content
  def does_device_own_content(self, device, content):
    return content in self.topology.vertex_properties['created'][device]

### CONTENT

  # Create a content
  # If the producer is not moving, creates the content, otherwise schedules for the next time step
  # First it creates the content
  # Then schedules to push the content
  def content_creation_actions(self, (content, source)):
    source = self.topology.vertex(source)
    if self.topology.vertex_properties['online'][source]:
      self.create_content(content, source)
      number_replicas = max(1, ((1 - min(1, ((self.current_session[int(source)][1] - (self.timestep-self.current_session[int(source)][0])) / float(self.capped_session)))) * self.max_replicas))
      self.content_push_actions((content, int(source), number_replicas))
    else:
      self.event_queue.add_event(self.timestep + 1, (EVENTS['CONTENT_CREATION'], (content, int(source))))

  # Generate a given content
  # Update the structures
  # Place the content
  def create_content(self, content, source):
    # Generate content
    self.contents[content] = {'creation': self.timestep, 'source': int(source), 'copies': []}
    # Place content
    self.place_content(content, source, True)

  # Place the content on a device and update some structures.
  def place_content(self, content, device, is_source=False, innetwork=False):
    device = self.topology.vertex(device)
    # Source verification, marks the device as the producer
    if is_source:
      self.topology.vertex_properties['created'][device].append(content)
    # Cache verification. Checks if the content is already in cache
    else:
      if not self.check_cache(device, content, innetwork):
        return False
    # Check if the content is being placed in a router or in a user device
    # If it is in a router, just place it in there
    # If it is in a user device (not the producer), then caches in the retrieve path
    if innetwork:
      self.cache_content(device, content)
    else:
      self.cache_content(device, content, is_device=True)

      if not is_source:
        self.add_innetwork_cache(content, self.contents[content]['source'], int(device))
        self.announce(device, content=content, pushed=True)

    return True

  # Announce a content object
  # If the content was pushed, the device got available or moved
  def announce(self, device, content=None, pushed=False, available=False):
    announced = False

    # If the content was pushed, announce it
    if pushed:
      self.topology.vertex_properties['announce'][device].append(content)
      announced = True

    # If the device moved or became available, check if there is any content to announce
    if available:
      for content in self.topology.vertex_properties['cache'][device]:
        if self.does_device_own_content(device, content) and self.is_device_home(device):
          pass
        elif content in self.topology.vertex_properties['announce'][device]:
          pass
        else:
          self.topology.vertex_properties['announce'][device].append(int(content))
          announced = True

    # Adds the cost to announce
    if announced:
      self.stats_announce_cost[content].append(int(device))

### STRATEGY

  # Push objects to other devices
  def content_push_actions(self, (content, source, max_copies)):
    source = self.topology.vertex(source)

    # If we are using the strategy, calculate it and the costs
    if max_copies > 0:
      self.push_content(content, source, max_copies)
    # If the producer does not push content, it just adds that there is no cost
    # It never executes this again
    else:
      if len(self.stats_pushing_time[content]) == 0:
        self.stats_pushing_time[content].append(0)
      if len(self.stats_vicinity_cost[content]) == 0:
        self.stats_vicinity_cost[content].append(0)

  # Push content to other devices. First it defines the pool of possible devices to push content to.
  # Then, it selects the best or random device(s) to push
  def push_content(self, content, source, max_copies):
    vicinity_devices = self.learn_vicinity(source, content)
    copies_placed = self.push_replicas(content, vicinity_devices, max_copies)

    # We could not push to everyone, try again next time step
    unpushed_copies = max_copies - copies_placed
    if unpushed_copies > 0:
      self.event_queue.add_event(self.timestep + 1, (EVENTS['CONTENT_PUSH'], (content, int(source), unpushed_copies)))

  def learn_vicinity(self, source, content):
    vicinity_devices = []

    # Get the vicinity
    vicinity_devices, vicinity_cost = self.get_vicinity(source, self.vicinity_size)

    # Set the cost to learn about the vicinity
    self.stats_vicinity_cost[content].append(vicinity_cost) #, len(vicinity_devices))

    # Rank the devices according to the placement policy
    vicinity_devices = self.rank_devices(vicinity_devices, content, self.placement_policy)

    return vicinity_devices

  def push_replicas(self, content, vicinity_devices, max_copies):
    # Attempts to push at most max_copies
    copies_placed = 0
    while copies_placed < max_copies and len(vicinity_devices) > 0:

      device = self.select_device(vicinity_devices, self.placement_policy)

      # A push will only fail if the device already had the content in cache
      if self.place_content(content, device, False):
        self.get_push_cost(content, device)
        copies_placed += 1

    return copies_placed

  # Get the push cost from the producer to a provider of a content
  def get_push_cost(self, content, provider):
    producer = self.topology.vertex(self.contents[content]['source'])
    self.stats_pushing_time[content].append(self.get_distance(self.topology.vertex_properties['position'][provider], self.topology.vertex_properties['position'][producer]) + 2)

  # Builds the set of devices from the source with vicinity distance
  # We go through the routers
  def get_vicinity(self, source, vicinity_distance):
    visit_queue = []
    visited = []
    vicinity = []
    cost = 0
    
    # Build neighborhood
    neighbors = source.all_neighbours()
    for node in neighbors:
      node = self.topology.vertex(node)
      if self.topology.vertex_properties['router'][node]:
        visit_queue.append((node, 1))
      else:
        if self.topology.vertex_properties['online'][node] and int(node) not in vicinity:
          vicinity.append(node)
          cost += 1

    while len(visit_queue) > 0:
      router, distance = visit_queue.pop(0)
      if int(router) not in visited and distance < vicinity_distance:
        cost += 1
        visited.append(int(router))
        neighbors = router.all_neighbours()
        for node in neighbors:
          node = self.topology.vertex(node)
          if self.topology.vertex_properties['router'][node]:
            visit_queue.append((node, distance+1))
          else:
            if self.topology.vertex_properties['online'][node] and int(node) not in vicinity:
              vicinity.append(node)
              cost += 1

    return vicinity, cost

  # Rank the devices according to some strategy
  def rank_devices(self, devices_set, content, strategy):
    device_rank = []

    if strategy == RANDOM_DEVICE:
      numpy.random.shuffle(devices_set)
      device_rank = devices_set

    else:
      for device in devices_set:
        metric_value = self.get_device_metric(device, strategy)
        # Adds a random value for tie-break
        device_rank.append((metric_value, numpy.random.random(), device))
      device_rank.sort(reverse=True)

    return device_rank

  # Get the device metric according to a strategy
  def get_device_metric(self, device, strategy):
    if strategy == BEST_AVAILABLE_DEVICE:
      return (self.current_session[int(device)][1] - (self.timestep+self.current_session[int(device)][0])) / float(self.capped_session)

  # Select the device to push according to the strategy
  def select_device(self, devices_set, strategy):
    device = None

    if self.placement_policy == RANDOM_DEVICE:
      device = devices_set.pop(0)

    elif self.placement_policy == BEST_AVAILABLE_DEVICE:
      metric_value, random_value, device = devices_set.pop(0)

    return device

### SEND REQUESTS

  # The actions to request a content
  # First, it calculates the request path and total retrieval time
  # Then, it requests the content and checks if it was successful or not
  def consumer_request_actions(self, (device, content)):
    device = self.topology.vertex(device)

    if content in self.contents:

      provider_list = self.get_available_copies(content)
      provider, retrieve_time = self.request_content(device, content, provider_list)

      self.retrieve_content(device, content, provider, retrieve_time, len(provider_list) == 0)

    else:
      self.event_queue.add_event(self.timestep + 1, (EVENTS['CONSUMER_REQUEST'], (int(device), content)))

  # Retrieve the content
  # If there are no available providers, it is a failure and attempts again next time step
  # Otherwise, it is a success! Adds in-network caching and refresh access to content in cache (for LRU)
  # Also collect some statistics
  def retrieve_content(self, device, content, provider, retrieve_time, failure):
    if int(device) not in self.stats_retrieval_time[content]:
      self.stats_retrieval_time[content][int(device)] = []
      self.stats_hit_rate[content][int(device)] = []

    # Adds the time, independent of success or failure
    self.stats_retrieval_time[content][int(device)].append(retrieve_time)

    if failure:
      self.stats_hit_rate[content][int(device)].append(0)
      self.event_queue.add_event(self.timestep + 1, (EVENTS['CONSUMER_REQUEST'], (int(device), content)))
    else:
      self.stats_hit_rate[content][int(device)].append(1)
      self.topology.vertex_properties['interest'][device][content] = 0
      self.refresh_cache_entry(provider, content)
      self.add_innetwork_cache(content, provider, int(device))

  # Get the providers for a content that are online at the moment
  def get_available_copies(self, content):
    device_list = []
    for device in self.contents[content]['copies']:
      device = self.topology.vertex(device)

      if self.topology.vertex_properties['online'][device]:
        device_list.append(device)
    return device_list

  # The announcement time is the minimum time for a provider's announcement to reach the request path to the producer
  # We consider the worst case scenario, that is the eccentricity of the provider's router + 1 (provider to router)
  def get_announce_time(self, provider, content):
    number_hops = 0

    # Only if the provider had to announce
    if content not in self.topology.vertex_properties['announce'][provider]:
      return number_hops

    provider_router = self.topology.vertex_properties['position'][provider]

    return self.eccentricity[provider_router] + 1

  # Find the closest provider from the consumer and estimate the retrieval time
  def request_content(self, device, content, provider_list):
    producer = self.topology.vertex(self.contents[content]['source'])

    device_router = self.topology.vertex_properties['position'][device]

    # Default request
    # The get_distance gets the distance between the routers. That is why we add 2 (the distance from the devices to the routers)
    # We also multiply it by 2 because we have to send the request and receive the content
    retrieve_time = 2 * (self.get_distance(device_router, self.topology.vertex_properties['position'][producer]) + 2)
    targets = [producer]

    # Check if there are closer providers
    for provider in provider_list:
      new_time = 2 * (self.get_distance(device_router, self.topology.vertex_properties['position'][provider]) + 2)
      if new_time < retrieve_time:
        retrieve_time = new_time
        targets = [provider]
      elif new_time == retrieve_time:
        targets.append(provider)

    # Check the smallest announce time
    # The idea is that the request time will be the announce time plus the retrieve time
    # Therefore, we need the lowest announce time for the closest provider
    requested_providers = []
    announce_time = 1000 # infinite

    for provider in targets:
      new_announce_time = self.get_announce_time(provider, content)
      if new_announce_time < announce_time:
        announce_time = new_announce_time
        requested_providers = [provider]
      elif new_announce_time == announce_time:
        requested_providers.append(provider)

    # Check in-network path
    # If we have multiple providers with the same time to retrieve, we just pick one at random
    # We then check whether some router in the path has the content cached.
    # It is important to notice that we (as consumers) are not aware of routing caching
    selected_provider = numpy.random.choice(requested_providers)
    selecter_provider_router = self.topology.vertex_properties['position'][selected_provider]

    selected_path = self.get_path(device_router, selecter_provider_router)
    if device_router != selected_path[0]:
      selected_path.reverse() # from the client towards the provider
    for router in selected_path:
      router = self.topology.vertex(router)
      if self.is_content_in_cache(router, content):
      # if content in self.topology.vertex_properties['cache'][router]:
        # We add only one here because the request is from a device to a router.
        # We still have to consider the consumer to the router, but there is no target device.
        retrieve_time = 2 * (self.get_distance(device_router, int(router)) + 1)
        selected_provider = router
        break

    total_retrive_time = announce_time + retrieve_time
    self.stats_announce_time[content].append(announce_time)

    return selected_provider, total_retrive_time

### CACHE

  # Timeout a cache entry
  def cache_timeout_actions(self, (router, content)):
    router = self.topology.vertex(router)
    self.remove_cached_content(router, content=content)

  # Caches a content in the path from source to device
  def add_innetwork_cache(self, content, source, device):
    source = self.topology.vertex(source)
    device = self.topology.vertex(device)

    source_router = self.topology.vertex_properties['position'][source]
    device_router = self.topology.vertex_properties['position'][device]

    for router in self.get_path(source_router, device_router):
      self.place_content(content, self.topology.vertex(router), innetwork=True)

  # Cache a content
  # If it is a user device, adds to the list of known providers
  # Otherwise (it is a router), schedules to timeout
  def cache_content(self, device, content, is_device=False):
    self.refresh_cache_entry(device, content)
    if is_device:
      self.contents[content]['copies'].append(int(device))
    else:
      self.event_queue.add_event(self.timestep + self.cache_timeout, (EVENTS['CACHE_TIMEOUT'], (int(device), content)))

  # Remove a cached content from the cache
  # Uses a LRU unless specified (timeout)
  def remove_cached_content(self, device, content=None, is_device=False):
    if content is not None:
      removed_content = content
    else:
      least_recently_used = 10000
      removed_content = -1
      for option in self.topology.vertex_properties['cache'][device]:
        if option not in self.topology.vertex_properties['created'][device]:
          if self.topology.vertex_properties['cache'][device][option] < least_recently_used:
            removed_content = option
            least_recently_used = self.topology.vertex_properties['cache'][device][option]

    if removed_content >= 0:
      if removed_content in self.topology.vertex_properties['cache'][device]:
        del self.topology.vertex_properties['cache'][device][removed_content]

      if is_device:
        self.contents[removed_content]['copies'].remove(int(device))

  # Check the cache
  # If the content is already there
  # If the cache is full
  def check_cache(self, device, content, innetwork=False):
    if self.is_content_in_cache(device, content):
      return False

    # Remove a content if cache is full
    if self.is_cache_full(device):
      if innetwork:
        self.remove_cached_content(device)
      else:
        self.remove_cached_content(device, is_device=True)
    return True

  # Access a cache entry, refreshing last time accessed
  def refresh_cache_entry(self, device, content):
    self.topology.vertex_properties['cache'][device][content] = self.timestep

  # Is the content in cache?
  def is_content_in_cache(self, device, content):
    return content in self.topology.vertex_properties['cache'][device]

  # Is the cache full?
  def is_cache_full(self, device):
    return len(self.topology.vertex_properties['cache'][device]) >= self.cache_size

### MOVEMENT 

  # Checks for each device if it is moving, sets to the new location
  def device_movement_actions(self, device):
    # reset announce. This is the first event of the time step
    self.topology.vertex_properties['announce'][device] = []
    move_device(device)

  # Move device. Disconnect from current router and selects a new one to connect
  def move_device(self, device):
    device_index = int(device)
    router_index = self.topology.vertex_properties['position'][device]
    
    # Remove current connection (location)
    edge = self.topology.edge(device_index, router_index)
    self.topology.remove_edge(edge)
    
    # Calculate the new position
    new_router_index = numpy.random.choice(self.topology.vertex_properties['places'][device])

    # Add new connection at new place
    self.topology.vertex_properties['position'][device] = new_router_index
    self.topology.add_edge(device_index, new_router_index)
    self.topology.vertex_properties['online'][device] = False

    distance = self.get_distance(router_index, new_router_index)
    speed_index = numpy.random.random()
    speed = self.user_speed_range[0] + (speed_index*(self.user_speed_range[1]-self.user_speed_range[0]))
    unavailable_period = max(1, distance / speed)

    self.event_queue.add_event(self.timestep + int(math.ceil(unavailable_period)), (EVENTS['START_SESSION'], device))
    
### END GAME

  def end_game(self):
    print "Version", VERSION

    total_hit_rate = []
    total_retrieve_time = []
    total_vicinity_cost = []
    total_pushing_time = []
    total_announce_time = []
    total_number_announcemnets = []
    total_replicas = []
    total_attempts = []

    for content in self.contents:
      # HIT RATE
      retrieve_attempts = 0
      retrieve_success = 0
      for device in self.stats_hit_rate[content]:
        retrieve_attempts += len(self.stats_hit_rate[content][device])
        retrieve_success += self.stats_hit_rate[content][device].count(1)
      hit_rate = retrieve_success / (max(1, float(retrieve_attempts)))
      total_hit_rate.append(hit_rate)

      # RETRIEVE TIME
      retrieve_times = []
      for device in self.stats_retrieval_time[content]:
        retrieve_times.append(sum(self.stats_retrieval_time[content][device]))
      average_retrieve_time = numpy.average(retrieve_times)
      total_retrieve_time.append(average_retrieve_time)

      # PUSHING TIME
      pushing_time = sum(self.stats_pushing_time[content])
      total_pushing_time.append(pushing_time)

      replicas = len(self.stats_pushing_time[content])
      total_replicas.append(replicas)

      # VICINITY COST
      vicinity_cost = sum(self.stats_vicinity_cost[content])
      total_vicinity_cost.append(vicinity_cost)

      attempts = len(self.stats_vicinity_cost[content])
      total_attempts.append(attempts)

      # ANNOUNCE TIME
      announce_time = numpy.average(self.stats_announce_time[content])
      total_announce_time.append(announce_time)

      # ANNOUNCE COST
      number_announcements = len(self.stats_announce_cost[content])
      total_number_announcemnets.append(number_announcements)

      print "Content", content, hit_rate, average_retrieve_time, pushing_time, vicinity_cost, announce_time, number_announcements, replicas, attempts

    print "Average", numpy.average(total_hit_rate), numpy.average(total_retrieve_time), numpy.average(total_pushing_time), numpy.average(total_vicinity_cost), numpy.average(total_announce_time), numpy.average(total_number_announcemnets), numpy.average(total_replicas), numpy.average(total_attempts)

    print 'Execution Time:', time.time() - begin_time

### SIMULATION ENGINE

  def execute(self):
    valid = True

    # Main loop
    while valid:
      # Get next event
      self.timestep = self.event_queue.next_timestep()
      events = self.event_queue.get_events(self.timestep)

      # Handle it
      for event, parameter in events:
        if event not in self.events_counter:
          self.events_counter[event] = 0
        self.events_counter[event] += 1

        if event == EVENTS['CONSUMER_REQUEST']:
          self.consumer_request_actions(parameter)
        elif event == EVENTS['CACHE_TIMEOUT']:
          self.cache_timeout_actions(parameter)
        elif event == EVENTS['START_SESSION']:
          self.start_session(parameter)
        elif event == EVENTS['START_MOVEMENT']:
          self.move_device(parameter)
        elif event == EVENTS['CONTENT_PUSH']:
          self.content_push_actions(parameter)
        elif event == EVENTS['CONTENT_CREATION']:
          self.content_creation_actions(parameter)
        elif event == EVENTS['END_TIME']:
          self.end_game()
          valid = False
      self.event_queue.remove_events(self.timestep)

simulator = Simulator(sys.argv[1])
simulator.setup_simulation()
simulator.execute()

