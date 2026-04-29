
# Statistics monitor actor
set MonitorActorStatistics_once_flag 0
proc MonitorActorStatistics {} {
	global STKO_VAR_process_id
	global STKO_VAR_increment
	global STKO_VAR_time_increment
	global STKO_VAR_time
	global STKO_VAR_num_iter
	global STKO_VAR_error_norm
	global STKO_VAR_percentage
	global MonitorActorStatistics_once_flag
	# Statistics
	if {$STKO_VAR_process_id == 0} {
		if {$MonitorActorStatistics_once_flag == 0} {
			set MonitorActorStatistics_once_flag 1
			set STKO_monitor_statistics [open "./STKO_monitor_statistics.stats"  w+]
		} else {
			set STKO_monitor_statistics [open "./STKO_monitor_statistics.stats"  a+]
		}
		puts $STKO_monitor_statistics "$STKO_VAR_increment $STKO_VAR_time_increment $STKO_VAR_time $STKO_VAR_num_iter $STKO_VAR_error_norm $STKO_VAR_percentage"
		close $STKO_monitor_statistics
	}
}
lappend STKO_VAR_MonitorFunctions "MonitorActorStatistics"

# Timing monitor actor
set monitor_actor_time_0 [clock seconds]
proc MonitorActorTiming {} {
	global monitor_actor_time_0
	global STKO_VAR_process_id
	if {$STKO_VAR_process_id == 0} {
		set STKO_time [open "./STKO_time_monitor.tim" w+]
		set current_time [clock seconds]
		puts $STKO_time $monitor_actor_time_0
		puts $STKO_time $current_time
		close $STKO_time
	}
}
lappend STKO_VAR_MonitorFunctions "MonitorActorTiming"

recorder mpco "results.mpco" \
-N "displacement" "rotation" "velocity" "angularVelocity" "acceleration" "angularAcceleration" "reactionForce" "reactionMoment" "reactionForceIncludingInertia" "reactionMomentIncludingInertia" "rayleighForce" "rayleighMoment" "unbalancedForce" "unbalancedMoment" "unbalancedForceIncludingInertia" "unbalancedMomentIncludingInertia" "pressure" "modesOfVibration" "modesOfVibrationRotational" \
-E "force" "deformation" "localForce" "damage" "equivalentPlasticStrain" "cw" "section.force" "section.deformation" "material.stress" "material.strain" "material.damage" "material.equivalentPlasticStrain" "material.cw" "section.fiber.stress" "section.fiber.strain" "section.fiber.damage" "section.fiber.equivalentPlasticStrain" "section.fiber.cw"

# Monitor Actor [2]
set nodes_X_2 {3}
set nodes_Y_2 {1 4}
set MonitorActor2_once_flag 0
proc MonitorActor2 {} {
	global MonitorActor2_once_flag
	global STKO_VAR_process_id
	global STKO_VAR_increment
	if {$MonitorActor2_once_flag == 0} {
		set MonitorActor2_once_flag 1
		set STKO_plot_00 [open "./STKO_plot_monitor2.plt" w+]
		puts $STKO_plot_00 "Displacement (X) 	Reaction Force (X) "
	} else {
		set STKO_plot_00 [open "./STKO_plot_monitor2.plt" a+]
	}
	reactions
	set monitor_value_X 0.0
	global nodes_X_2
	foreach node_id $nodes_X_2 {
		# get node value
		set node_value [nodeDisp $node_id 1]
		set monitor_value_X [expr $monitor_value_X + $node_value]
	}
	set monitor_value_X [expr 1.0 * $monitor_value_X + 0.0]
	set monitor_value_Y 0.0
	global nodes_Y_2
	foreach node_id $nodes_Y_2 {
		# get node value
		set node_value [nodeReaction $node_id 1]
		set monitor_value_Y [expr $monitor_value_Y + $node_value]
	}
	set monitor_value_Y [expr 1.0 * $monitor_value_Y + 0.0]
	puts $STKO_plot_00 "$monitor_value_X	$monitor_value_Y"
	close $STKO_plot_00
}
lappend STKO_VAR_MonitorFunctions "MonitorActor2"

# Constraints.sp fix
	fix 1 1 1 1 1 1 1
	fix 4 1 1 1 1 1 1

# Patterns.addPattern loadPattern
pattern Plain 4 1 -fact 180000.0 {

# Loads.Force NodeForce
	load 2 1.0 0.0 0.0 0.0 0.0 0.0
}

# analyses command
domainChange
constraints Transformation
numberer RCM
system UmfPack
test NormDispIncr 0.01 200  
algorithm KrylovNewton
integrator LoadControl 0.0
analysis Static
# ======================================================================================
# NON-ADAPTIVE LOAD CONTROL ANALYSIS
# ======================================================================================

# ======================================================================================
# USER INPUT DATA 
# ======================================================================================

# duration and initial time step
set total_duration 1.0
set initial_num_incr 200

set STKO_VAR_time 0.0
set STKO_VAR_time_increment [expr $total_duration / $initial_num_incr]
set STKO_VAR_initial_time_increment $STKO_VAR_time_increment
integrator LoadControl $STKO_VAR_time_increment 
for {set STKO_VAR_increment 1} {$STKO_VAR_increment <= $initial_num_incr} {incr STKO_VAR_increment} {
	
	# before analyze
	STKO_CALL_OnBeforeAnalyze
	
	# perform this step
	set STKO_VAR_analyze_done [analyze 1 ]
	
	# update common variables
	if {$STKO_VAR_analyze_done == 0} {
		set STKO_VAR_num_iter [testIter]
		set STKO_VAR_time [expr $STKO_VAR_time + $STKO_VAR_time_increment]
		set STKO_VAR_percentage [expr $STKO_VAR_time/$total_duration]
		set norms [testNorms]
		if {$STKO_VAR_num_iter > 0} {set STKO_VAR_error_norm [lindex $norms [expr $STKO_VAR_num_iter-1]]} else {set STKO_VAR_error_norm 0.0}
	}
	
	# after analyze
	set STKO_VAR_afterAnalyze_done 0
	STKO_CALL_OnAfterAnalyze
	
	# check convergence
	if {$STKO_VAR_analyze_done == 0} {
		# print statistics
		if {$STKO_VAR_process_id == 0} {
			puts [format "Increment: %6d | Iterations: %4d | Norm: %8.3e | Progress: %7.3f %%" $STKO_VAR_increment $STKO_VAR_num_iter  $STKO_VAR_error_norm [expr $STKO_VAR_percentage*100.0]]
		}
	} else {
		# stop analysis
		error "ERROR: the analysis did not converge"
	}
	
}

# done
if {$STKO_VAR_process_id == 0} {
	puts "Target time has been reached. Current time = $STKO_VAR_time"
	puts "SUCCESS."
}
wipeAnalysis

# Done!
puts "ANALYSIS SUCCESSFULLY FINISHED"
