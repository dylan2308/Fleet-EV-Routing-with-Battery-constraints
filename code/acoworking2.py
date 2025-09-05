# -------------------------------------------
# ACO for the EV Routing Problem (single-depot, hard battery & time windows)
# -------------------------------------------
import numpy as np
import pandas as pd
import random
from collections import deque
from config import *
import json
import time

# ---------- USER-SIDE INPUTS ---------------
# distance_matrix      (n×n numpy array)  – already created
# energy_matrix        (n×n numpy array) 
# time_matrix          (n×n numpy array)
# customer_nodes       (list of indices 1..C)
# charger_nodes        (list of indices)
# BATTERY_CAPACITY     (kWh)
# TOTAL_TIME_ALLOWED   (min)

random.seed(SEED)

# Electric Vehicle Points
ev_pts = pd.read_csv("Unique_EV_Points.csv")
charge_pts = ev_pts.sample(n=NUM_CHARGERS, random_state=SEED)[['Longitude', 'Latitude']] # dataframe
charge_pts_list = list(charge_pts.itertuples(index=False, name=None)) # list

# Delivery locations Points
dl = pd.read_csv("DeliveryLocations.csv")
todays_locations = dl.sample(n=NUM_CUSTOMERS, random_state=SEED)
dl_list = list(todays_locations.itertuples(index=False, name=None))

# Fedex Ship Centre
fedex_centre = [(104.0023106, 1.3731437)]

locations = fedex_centre + dl_list + charge_pts_list # combined list of all locations

# Total nodes: 1 depot + customers + chargers
n = len(locations)
customer_nodes = list(range(1, 1 + NUM_CUSTOMERS)) # gets the index/nodes of customers
charger_nodes = list(range(1 + NUM_CUSTOMERS, n))  

# Compute Distance & Energy Matrices
distance_matrix = np.zeros((n, n))
time_matrix = np.zeros((n, n))
energy_matrix = np.zeros((n, n))

def calc_dist(p1, p2):
    """
    Calculates the straight-line (euclidean) distance
    """
    lon_diff = p1[0] - p2[0]
    lat_diff = p1[1] - p2[1]
    return 111 * np.sqrt(lon_diff**2 + lat_diff**2)

for i in range(n):
    for j in range(n):
        if i != j:
            dist = calc_dist(locations[i], locations[j])
            distance_matrix[i][j] = dist
            energy_matrix[i][j] = dist * ENERGY_PER_KM
            time_matrix[i][j] = dist * TIME_PER_KM

# -----------------------------------------------------------------
#  Escape‑feasibility pre‑computation  (minimum energy/time needed
#  to reach *some* charger or the depot from every node)
# -----------------------------------------------------------------
ESCAPE_ENERGY = np.zeros(n)
ESCAPE_TIME   = np.zeros(n)
depot = 0
escape_targets = [depot] + charger_nodes            # depot + chargers
for v in range(n):
    ESCAPE_ENERGY[v] = min(energy_matrix[v][c] for c in escape_targets)
    ESCAPE_TIME[v]   = min(time_matrix[v][c] + time_matrix[c][depot] for c in escape_targets)

def total_distance(routes, locations):
    """
    Returns total distance of for all route in routes
    """
    dist = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            dist += calc_dist(locations[route[i]], locations[route[i+1]])
    return dist

def num_activated_vehicles(routes):
    """
    Calculates number of activated vehicles
    """
    return len(routes) - sum(1 for route in routes if route == [0, 0])


# ANT_PARAMS ---------- tweak here ----------
NUM_ANTS      = 500              # usually ≈ #customers
MAX_ITERS     = 200             # stopping criterion
ALPHA         = 0.5             # pheromone weight
BETA          = 2.0             # heuristic (1/dist) weight
RHO           = 0.1             # evaporation rate
Q             = 1.0             # deposit factor
ELITE_RANK    = 3.0
# -------------------------------------------

# ------------------------------------------------------------------
# Helper functions for external hyper‑parameter tuning
# ------------------------------------------------------------------
def sample_hyperparams():
    """
    Draw a random hyper‑parameter set for ACO.  The ranges are coarse
    but wide enough for rough tuning.
    """
    num_cust = len(customer_nodes)
    return {
        "ALPHA": np.random.uniform(0.1, 2.0),
        "BETA":  np.random.uniform(1.0, 5.0),
        "RHO":   np.random.uniform(0.05, 0.1),
        "Q":     np.random.uniform(0.1, 3.0),
        "NUM_ANTS": np.random.randint(max(10, num_cust // 2), num_cust + 1),
        "ELITE_RANK": np.random.randint(1, 4)
    }

def run_aco_once(params, *, max_iters=300, cpu_limit=60):
    """
    Run ACO once with the supplied hyper‑parameter dictionary.
    Returns (best_distance, cpu_time_seconds).

    The global data structures (distance_matrix, energy_matrix, …)
    are reused, but pheromone and best‑cost logs are local to this run
    so consecutive calls are independent.
    """
    # Inject params into module namespace so helper routines pick them up
    globals().update(params)
    global eta
    eta = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)

    pher = np.full((n, n), 1e-6)      # fresh pheromone matrix
    best = float("inf")
    start = time.time()
    it = 0

    while it < max_iters and (time.time() - start) < cpu_limit:
        it += 1
        sols, costs = [], []

        for _ in range(NUM_ANTS):
            unrouted = set(customer_nodes)
            routes   = []
            while unrouted:
                route   = [depot]
                battery = BATTERY_CAPACITY
                t_used  = 0.0
                cur     = depot
                while True:
                    nxt = pick_next(cur, unrouted, pher, eta, battery, t_used)
                    if nxt is None:
                        break
                    route.append(nxt)
                    battery -= energy_matrix[cur][nxt]
                    t_used  += time_matrix[cur][nxt]
                    cur      = nxt
                    unrouted.remove(nxt)
                route.append(depot)
                routes.append(route)

            cost = sum(distance_matrix[i][j]
                       for r in routes for i, j in zip(r, r[1:]))
            sols.append(routes)
            costs.append(cost)
            best = min(best, cost)

        # --- pheromone update ---
        pher *= (1.0 - RHO)           # evaporation
        for idx in np.argsort(costs)[:ELITE_RANK]:
            deposit = Q / max(costs[idx], 1e-9)
            for r in sols[idx]:
                for i, j in zip(r, r[1:]):
                    pher[i][j] += deposit
                    pher[j][i] += deposit     # symmetric

    return best, time.time() - start

# ── Stopping‑rule toggles ─────────────────────────
USE_TIME_LIMIT = True    # True ⇒ stop by CPU‑time, False ⇒ stop by iterations
MAX_CPU_TIME   = 120       # seconds (used only when USE_TIME_LIMIT = True)
# ──────────────────────────────────────────────────

# n = distance_matrix.shape[0]     # depot + customers + chargers
# depot = 0

# ------- 1.  helper: choose next node -----------------
def pick_next(current, allowed, pher, eta, battery, time):
    """Return the next feasible node (roulette-wheel)."""
    # filter allowed list by feasibility
    feasible = []
    probs    = []
    for j in allowed:
        e_need = energy_matrix[current][j]
        t_need = time_matrix[current][j]

        # battery/time after arriving at j
        b_after = battery - e_need
        t_after = time  + t_need

        # ------------- look‑ahead feasibility test -------------
        if b_after < 0:
            continue                              # cannot even reach j
        if b_after < ESCAPE_ENERGY[j]:
            continue                              # would be stranded at j
        if t_after + ESCAPE_TIME[j] > TOTAL_TIME_ALLOWED:
            continue                              # no time left to escape
        # --------------------------------------------------------

        feasible.append(j)
        probs.append((pher[current][j] ** ALPHA) * (eta[current][j] ** BETA))

    if not feasible:                      # dead end
        return None
    probs = np.array(probs)
    probs /= probs.sum()
    return np.random.choice(feasible, p=probs)

# ------- 2.  main ACO loop ----------------------------

def run_aco_main():
    pheromone = np.full((n, n), 1e-6)         # tiny τ0 > 0
    # “visibility” = inverse distance, avoid 1/0 on diagonal
    # avoid divide‑by‑zero on the diagonal
    global eta
    eta = np.where(distance_matrix > 0,
                   1.0 / distance_matrix,
                   0.0)

    global best_cost, best_solution, best_distance_log, time_log
    best_cost     = float('inf')
    best_distance_log = []
    time_log = []
    import time
    best_solution = None

    start_time = time.time()
    it = 0

    while True:                           # dynamic stopping rule
        it += 1

        ant_solutions = []
        ant_costs     = []

        # ---- 2.1  each ant constructs a full set of routes ----
        for ant in range(NUM_ANTS):
            unrouted = set(customer_nodes)
            routes   = []                     # list of lists

            while unrouted:
                route     = [depot]
                battery   = BATTERY_CAPACITY
                time_sofar = 0.0
                current   = depot

                while True:
                    nxt = pick_next(current, unrouted, pheromone, eta,
                                    battery, time_sofar)
                    if nxt is None:
                        # no feasible customer ⇒ try to recharge or finish
                        # (a) if nearest charger lets us serve more later
                        recharge   = min(
                            charger_nodes,
                            key=lambda c: distance_matrix[current][c])
                        if (battery < BATTERY_CAPACITY and
                            time_sofar + time_matrix[current][recharge] +
                            time_matrix[recharge][depot] <= TOTAL_TIME_ALLOWED):
                            # go recharge
                            route.append(recharge)
                            time_sofar += time_matrix[current][recharge]
                            battery = BATTERY_CAPACITY
                            current = recharge
                            continue       # try again to pick customer
                        else:
                            break          # return to depot

                    # ------------- battery check -------------
                    e_need = energy_matrix[current][nxt]
                    b_after = battery - e_need
                    # extra safety: stranded check
                    if b_after < 0 or b_after < ESCAPE_ENERGY[nxt]:
                        # nearest reachable charger
                        nearest_charger = min(
                            charger_nodes,
                            key=lambda c: distance_matrix[current][c]
                        )
                        e_to_c = energy_matrix[current][nearest_charger]
                        t_to_c = time_matrix[current][nearest_charger]

                        # can we physically reach the charger and still get home on time?
                        if (battery >= e_to_c and
                            time_sofar + t_to_c + time_matrix[nearest_charger][depot] <= TOTAL_TIME_ALLOWED):
                            # drive to charger (consume energy & time)
                            route.append(nearest_charger)
                            battery  -= e_to_c
                            time_sofar += t_to_c

                            # full recharge
                            battery = BATTERY_CAPACITY
                            current = nearest_charger
                            continue            # retry customer selection
                        else:
                            break               # cannot continue this route

                    # go to customer nxt
                    route.append(nxt)
                    battery   -= energy_matrix[current][nxt]
                    time_sofar += time_matrix[current][nxt]
                    current    = nxt
                    unrouted.remove(nxt)

                # ----- ensure we can go back to depot -----
                e_home = energy_matrix[current][depot]
                if battery < e_home:
                    # attempt detour to nearest charger
                    recharge = min(charger_nodes,
                                   key=lambda c: distance_matrix[current][c])
                    e_to_c   = energy_matrix[current][recharge]
                    t_to_c   = time_matrix[current][recharge]
                    if (battery >= e_to_c and
                        time_sofar + t_to_c + time_matrix[recharge][depot]
                            <= TOTAL_TIME_ALLOWED):
                        # insert recharge stop
                        route.append(recharge)
                        time_sofar += t_to_c
                        battery = BATTERY_CAPACITY
                        current = recharge
                    else:
                        # cannot reach a charger or depot with remaining battery/time
                        # → close the current vehicle’s tour at the depot and let the
                        #   outer `while unrouted:` loop launch a fresh vehicle
                        time_sofar += time_matrix[current][depot]   # (idealised direct return)
                        route.append(depot)
                        battery = 0.0                               # vehicle is out of juice
                        current = depot
                        # do *not* break the outer loop – we still have unrouted customers
                # close the route
                if current != depot:
                    # before final route.append(depot)
                    if time_sofar + time_matrix[current][depot] > TOTAL_TIME_ALLOWED:
                        break  # abandon this route
                    time_sofar += time_matrix[current][depot]
                    route.append(depot)
                routes.append(route)

            # ---- 2.2  compute cost & store  ----
            # skip empty routes to avoid zero-cost solutions
            if not routes:          # ant aborted ⇒ ignore this iteration
                continue

            # after all routes for this ant have been built, check coverage
            if unrouted:                     # some customers were left unserved
                # mark solution as very bad but keep it (so pheromone can learn)
                cost = 1e6 + len(unrouted)   # huge fixed penalty + tiny tie‑break
                ant_solutions.append(routes)
                ant_costs.append(cost)
                continue                     # skip normal cost computation
            else:
                cost = sum(distance_matrix[i][j]
                           for route in routes
                           for i, j in zip(route, route[1:]))
                ant_solutions.append(routes)
                ant_costs.append(cost)
                if cost < best_cost:
                    best_cost, best_solution = cost, routes

        # ---- 2.3  pheromone evaporation ----
        pheromone *= (1.0 - RHO)

        # ---- 2.4  pheromone deposit by top ants ----
        elite_idx = np.argsort(ant_costs)[:ELITE_RANK]
        for idx in elite_idx:
            routes = ant_solutions[idx]
            # skip ants that produced an empty or zero‑cost solution
            if ant_costs[idx] <= 1e-9:
                continue
            deposit = Q / ant_costs[idx]
            for route in routes:
                for i, j in zip(route, route[1:]):
                    pheromone[i][j] += deposit
                    pheromone[j][i] += deposit   # symmetric

        # ---- optional: print progress ----
        if (it % 20 == 0) or it == 1:
            print(f"Iter {it:3d}: best so far = {best_cost:.2f} km")

        # record progress for benchmarking
        time_log.append(time.time() - start_time)
        best_distance_log.append(best_cost)

        # ---------- stopping condition ----------
        if (not USE_TIME_LIMIT and it >= MAX_ITERS) or \
           (USE_TIME_LIMIT and (time.time() - start_time) >= MAX_CPU_TIME):
            break

    # ────────────────────────────────────────────────────────────────
    #  FEASIBILITY CHECK FOR ACO BEST SOLUTION
    # ────────────────────────────────────────────────────────────────
    def check_feasible(routes):
        """
        routes: list of routes  e.g. [[0, 3, 7, 0], [0, 5, 9, 0], …]
        Prints a report and returns True/False.
        """
        OK = True
        print("\n────────────  FEASIBILITY REPORT  ────────────")

        # 1) Customer coverage (each visited exactly once)
        visited = []
        for r in routes:
            visited += [n for n in r if n in customer_nodes]
        missing   = set(customer_nodes) - set(visited)
        repeated  = [c for c in set(visited) if visited.count(c) > 1]

        if missing:
            print("❌ Missing customers :", sorted(missing))
            OK = False
        else:
            print("✓ All customers visited once")

        if repeated:
            print("❌ Repeated customers:", sorted(repeated))
            OK = False

        # 2) Reverse-arc duplicates  (i,j) & (j,i) used by diff vehicles
        arc_owner = {}
        rev_conf  = []
        for k, route in enumerate(routes, 1):
            for i, j in zip(route, route[1:]):
                if (j, i) in arc_owner and arc_owner[(j, i)] != k:
                    rev_conf.append((i, j, k, arc_owner[(j, i)]))
                arc_owner[(i, j)] = k
        if rev_conf:
            print("❌ Reverse-arc duplicates:")
            for i, j, k1, k2 in rev_conf:
                print(f"   ({i},{j}) in veh {k1}  &  ({j},{i}) in veh {k2}")
            OK = False
        else:
            print("✓ No reverse-arc duplicates")

        # 3) Per‑vehicle battery & time trace  + early dip report
        for k, route in enumerate(routes, 1):
            batt     = BATTERY_CAPACITY
            t_tot    = 0.0
            feasible = True

            for i, j in zip(route, route[1:]):
                # recharge whenever the vehicle *leaves* a charger
                if i in charger_nodes:
                    batt = BATTERY_CAPACITY

                e_need = energy_matrix[i][j]
                t_need = time_matrix[i][j]

                # accumulate
                batt  -= e_need
                t_tot += t_need

                # detect first negative battery dip
                if batt < -1e-6:
                    print(f"⚠️  Veh {k}: battery negative after edge "
                          f"({i}→{j})  – needed {e_need:.2f}, "
                          f"had {batt + e_need:.2f}")
                    feasible = False          # keep evaluating to show all issues

            if t_tot > TOTAL_TIME_ALLOWED + 1e-6:
                feasible = False

            status = "OK" if feasible else "FAIL"
            print(f"Veh {k:>2}: time {t_tot:6.1f} /{TOTAL_TIME_ALLOWED}  "
                  f"| final batt {batt:6.2f}  → {status}")
            OK &= feasible

        print("──────────────  END REPORT  ──────────────")
        return OK

    # Call it on ACO best:
    isOK = check_feasible(best_solution)
    print("\nOVERALL:", "✓ Solution is feasible" if isOK else "❌ Infeasible")

    # ----------------  DONE  ------------------------------
    print("\n=== BEST FOUND SOLUTION ===")
    for r, route in enumerate(best_solution, 1):
        print(f"Veh {r}: {route}")
    print(f"Total distance = {best_cost:.2f} km")

    # ────────────────────────────────────────────────────────────────
    #  QUICK REACHABILITY AUDIT  – “Can one fresh van serve node j
    #  alone and still get home within TOTAL_TIME_ALLOWED?”
    # ────────────────────────────────────────────────────────────────
    def audit_reachability():
        # depot already defined above
        impossible = []     # customers that break the 300-min duty window
        tight      = []     # feasible but slack < 5 min

        for j in customer_nodes:
            t_out  = time_matrix[depot][j]            # depot → customer
            t_back = ESCAPE_TIME[j] + time_matrix[depot][depot]  # j → (charger/depot) → depot
            total  = t_out + t_back

            if total > TOTAL_TIME_ALLOWED + 1e-6:
                impossible.append((j, total))
            elif total > TOTAL_TIME_ALLOWED - 5:      # < 5 min slack
                tight.append((j, total))

        print("\n────────  REACHABILITY AUDIT  ────────")
        if impossible:
            print(f"❌  {len(impossible)} customers are unreachable in "
                  f"{TOTAL_TIME_ALLOWED} min:")
            for j, tot in impossible:
                print(f"   • node {j:>2}: solo trip needs {tot:6.1f} min")
        else:
            print("✓ Every customer *can* be served within "
                  f"{TOTAL_TIME_ALLOWED} min by a solo trip.")

        if tight:
            print(f"\n⚠️  {len(tight)} customers leave < 5 min slack "
                  "(feasible but very tight):")
            for j, tot in tight:
                print(f"   • node {j:>2}: solo trip needs {tot:6.1f} min")
        print("────────────────────────────────────────")

    # ---- call it once, right before the ACO main loop -------------
    audit_reachability()

    data = {
        'run': "ACO",
        'costs': best_distance_log,
        'cpu_times': time_log
    }

    with open('Benchmark_25_10_3.json', 'a') as file:
        file.write(json.dumps(data) + '\n')

# ------------------------------------------------------------------
# Stand‑alone execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_aco_main()