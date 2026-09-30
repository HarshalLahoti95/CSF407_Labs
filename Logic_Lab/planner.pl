% planner.pl -- warehouse knowledge base and an independent move verifier.

% ---- Task 6: facts describing which locations are connected ----
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% A rule: the robot can move between connected locations.
% This is the Prolog form of  Connected(X,Y) -> CanMove(X,Y).
can_move(X,Y) :- connected(X,Y).

% ---- Task 7: the verifier used to check a proposed plan step ----
valid_move(X,Y) :- connected(X,Y).

% Reachability, to show the difference between a one-step rule and its
% transitive closure: valid_move(a,c) fails, but a->c IS reachable via b.
reachable(X,Y) :- connected(X,Y).
reachable(X,Y) :- connected(X,Z), connected(Z,Y).

% ---- Task 8: a separate small example of chained inference ----
wet_road.
slippery     :- wet_road.
reduce_speed :- slippery.
