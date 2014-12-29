//
//  main.c
//  renewal_spiking
//
//  Created by Matthew Crossley on 11/1/12.
//  Copyright (c) 2012 Matthew Crossley. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// files
FILE *response_file;
FILE *response_time_file;
FILE *acc_file;
FILE *predicted_feedback_file;
FILE *dopamine_file;
FILE *correlation_file;
FILE *confidence_file;

FILE *vis_file;
FILE *w_vis_msn_A_file;
FILE *w_vis_msn_B_file;
FILE *w_vis_msn_C_file;
FILE *w_vis_msn_D_file;

FILE *w_pf_tan_file;
FILE *pf_file;

FILE *tan_output_file;
FILE *msn_output_A_file;
FILE *msn_output_B_file;
FILE *msn_output_C_file;
FILE *msn_output_D_file;
FILE *motor_output_A_file;
FILE *motor_output_B_file;
FILE *motor_output_C_file;
FILE *motor_output_D_file;

FILE *tan_v_file;
FILE *msn_v_A_file;
FILE *msn_v_B_file;
FILE *msn_v_C_file;
FILE *msn_v_D_file;
FILE *motor_v_A_file;
FILE *motor_v_B_file;
FILE *motor_v_C_file;
FILE *motor_v_D_file;

// model parameters
int num_acquisition_trials;
int num_extinction_trials;
int num_reacquisition_trials;
int num_trials;
float num_simulations;
int num_steps;
int T;
int tau;

int stim_onset;
int stim_duration;

int dim;
float vis_amp;
float vis_width;
float vis_dist_x;
float vis_dist_y;

float lateral_inhibition_msn;
float lateral_inhibition;

float w_noise_tan_L;
float w_noise_msn_L;
float w_noise_tan_U;
float w_noise_msn_U;
float noise_tan_mu;
float noise_tan_sigma;
float noise_msn_mu;
float noise_msn_sigma;
float noise_motor_mu;
float noise_motor_sigma;
float noise;
float resp_thresh;

float w_ltp_vis_msn;
float w_ltd_vis_msn_1;
float w_ltd_vis_msn_2;
float w_ltp_pf_tan;
float w_ltd_pf_tan_1;
float w_ltd_pf_tan_2;
float nmda;
float ampa;

float confidence;
int response;
int response_time;
float *outputs;
float max_output;

float obtained_feedback;
float predicted_feedback;
float w_prediction_error;
float da;
float da_base;
float da_alpha;
float da_beta;

float r_theta;
float *r_p_pos;
float *r_p_neg;
float *r_I_pos;
float *r_I_neg;
float *r_omega_pos;
float *r_omega_neg;
float *r_p_pos_mean;
float *r_p_neg_mean;
float correlation;

float **pf_sum;
float tan_sum;
float *vis_sum;
float msn_sum_A;
float msn_sum_B;
float msn_sum_C;
float msn_sum_D;

int num_pf_cells;
int num_pf_cells_per_context;
int num_contexts;
int num_pf_overlap;
int current_context;

float spike_a;
float spike_b;
float spike_length;

float pause_decay;
float pf_amp;
float pause_mod_amp;
float pf_tan_mod;

float w_vis_msn;
float w_msn_motor;
float w_tan_msn;
float **w_pf_tan;

float *w_vis_msn_A;
float *w_vis_msn_B;
float *w_vis_msn_C;
float *w_vis_msn_D;

float vis_act_A;
float vis_act_B;
float vis_act_C;
float vis_act_D;

float *vis_msn_act_A;
float *vis_msn_act_B;
float *vis_msn_act_C;
float *vis_msn_act_D;

// step records
float *spike;
float **stim;
float *vis;
float **pf;
float *pf_mod;
float *pause_mod;
float *pf_tan_act;

float *tan_v;
float *tan_u;
int *tan_spikes;
float *tan_output;

float *msn_v_A;
float *msn_u_A;
int *msn_spikes_A;
float *msn_output_A;

float *msn_v_B;
float *msn_u_B;
int *msn_spikes_B;
float *msn_output_B;

float *msn_v_C;
float *msn_u_C;
int *msn_spikes_C;
float *msn_output_C;

float *msn_v_D;
float *msn_u_D;
int *msn_spikes_D;
float *msn_output_D;

float *motor_v_A;
int *motor_spikes_A;
float *motor_output_A;

float *motor_v_B;
int *motor_spikes_B;
float *motor_output_B;

float *motor_v_C;
int *motor_spikes_C;
float *motor_output_C;

float *motor_v_D;
int *motor_spikes_D;
float *motor_output_D;

float confidence;
int response;
float *outputs;
float max_output;

float obtained_feedback;
float predicted_feedback;
float w_prediction_error;
float da;
float da_base;
float da_alpha;
float da_beta;

float r_theta;
float *r_p_pos;
float *r_p_neg;
float *r_I_pos;
float *r_I_neg;
float *r_omega_pos;
float *r_omega_neg;
float *r_p_pos_mean;
float *r_p_neg_mean;
float correlation;

// trial records
float **vis_record;
float **pf_record;
float **tan_v_record;
float **tan_output_record;
float **msn_v_A_record;
float **msn_v_B_record;
float **msn_v_C_record;
float **msn_v_D_record;
float **msn_output_A_record;
float **msn_output_B_record;
float **msn_output_C_record;
float **msn_output_D_record;
float **motor_v_A_record;
float **motor_v_B_record;
float **motor_v_C_record;
float **motor_v_D_record;
float **motor_output_A_record;
float **motor_output_B_record;
float **motor_output_C_record;
float **motor_output_D_record;

float **w_pf_tan_record;
float **w_vis_msn_A_record;
float **w_vis_msn_B_record;
float **w_vis_msn_C_record;
float **w_vis_msn_D_record;

float *response_record;
float *response_time_record;
float *predicted_feedback_record;
float *dopamine_record;
float *correlation_record;
float *confidence_record;
float *accuracy_record;

// average records
float **w_pf_tan_record_ave;
float **w_vis_msn_A_record_ave;
float **w_vis_msn_B_record_ave;
float **w_vis_msn_C_record_ave;
float **w_vis_msn_D_record_ave;

float *response_record_ave;
float *response_time_record_ave;
float *predicted_feedback_record_ave;
float *dopamine_record_ave;
float *correlation_record_ave;
float *confidence_record_ave;
float *accuracy_record_ave;

// chunking functions
void simulate_acquisition();
void simulate_intervention_nc_25();
void simulate_intervention_nc_40();
void simulate_intervention_nc_63();
void simulate_intervention_mixed_7525();
void simulate_reacquisition();
void simulate_nc_25();
void simulate_nc_40();
void simulate_nc_63();
void simulate_mixed_7525();


// funtions
void set_params();
void allocate_buffers();
void load_stim();
void init_weights();
void init_buffers();
void update_pf(int trial);
void update_vis(int trial);
void update_tan(int step);
void update_msn(int step);
void update_motor(int step);
void update_response(int step);
void update_feedback_contingent(int trial);
void update_feedback_nc_25(int trial);
void update_feedback_nc_40(int trial);
void update_feedback_nc_63(int trial);
void update_feedback_mixed_7525(int trial);
void update_dopamine_rpe(int trial);
void update_dopamine_corr(int trial);
void update_pf_tan(int trial);
void update_vis_msn(int trial);
void record_trial(int trial);
void record_simulation(int simulation);
void reset_trial();
void reset_simulation();
void reset_averages();
void shuffle_stimuli();
void compute_average_results();
void write_data_file(int path);

// utilities
float pos(float val);
float cap(float val);
float randn(float mean, float variance);

int main(int argc, const char * argv[])
{
    srand ( (float) time(NULL) );
    
    set_params();
    allocate_buffers();
    load_stim();
    init_weights();
    init_buffers();
    simulate_nc_25();
    simulate_nc_63();
    simulate_nc_40();
    simulate_mixed_7525();
    printf("finished\n");
    return 0;
}

void simulate_acquisition()
{
    for(int i=0; i<num_acquisition_trials; i++)
    {
        update_pf(i);
        update_vis(i);
        
        for(int t=0; t<num_steps; t++)
        {
            update_tan(t);
            update_msn(t);
            update_motor(t);
            update_response(t);
            
            if(response != -1)
            {
                response_time = t;
                
                for(int tt=response_time; tt<num_steps; tt++)
                {
                    update_tan(tt);
                    update_msn(tt);
                    update_motor(tt);
                }
                break;
            }
        }
        
        update_feedback_contingent(i);
        update_dopamine_corr(i);
        update_pf_tan(i);
        update_vis_msn(i);
        
        record_trial(i);
        reset_trial();
    }
}

void simulate_intervention_nc_25()
{
    for(int i=num_acquisition_trials; i<num_acquisition_trials+num_extinction_trials; i++)
    {
        update_pf(i);
        update_vis(i);
        
        for(int t=0; t<num_steps; t++)
        {
            update_tan(t);
            update_msn(t);
            update_motor(t);
            update_response(t);
            
            if(response != -1)
            {
                response_time = t;
                
                for(int tt=response_time; tt<num_steps; tt++)
                {
                    update_tan(tt);
                    update_msn(tt);
                    update_motor(tt);
                }
                break;
            }
        }
        
        update_feedback_nc_25(i);
        update_dopamine_corr(i);
        update_pf_tan(i);
        update_vis_msn(i);
        
        record_trial(i);
        reset_trial();
    }
}

void simulate_intervention_nc_40()
{
    for(int i=num_acquisition_trials; i<num_acquisition_trials+num_extinction_trials; i++)
    {
        update_pf(i);
        update_vis(i);
        
        for(int t=0; t<num_steps; t++)
        {
            update_tan(t);
            update_msn(t);
            update_motor(t);
            update_response(t);
            
            if(response != -1)
            {
                response_time = t;
                
                for(int tt=response_time; tt<num_steps; tt++)
                {
                    update_tan(tt);
                    update_msn(tt);
                    update_motor(tt);
                }
                break;
            }
        }
        
        update_feedback_nc_40(i);
        update_dopamine_corr(i);
        update_pf_tan(i);
        update_vis_msn(i);
        
        record_trial(i);
        reset_trial();
    }
}

void simulate_intervention_nc_63()
{
    for(int i=num_acquisition_trials; i<num_acquisition_trials+num_extinction_trials; i++)
    {
        update_pf(i);
        update_vis(i);
        
        for(int t=0; t<num_steps; t++)
        {
            update_tan(t);
            update_msn(t);
            update_motor(t);
            update_response(t);
            
            if(response != -1)
            {
                response_time = t;
                
                for(int tt=response_time; tt<num_steps; tt++)
                {
                    update_tan(tt);
                    update_msn(tt);
                    update_motor(tt);
                }
                break;
            }
        }
        
        update_feedback_nc_63(i);
        update_dopamine_corr(i);
        update_pf_tan(i);
        update_vis_msn(i);
        
        record_trial(i);
        reset_trial();
    }
}

void simulate_intervention_mixed_7525()
{
    for(int i=num_acquisition_trials; i<num_acquisition_trials+num_extinction_trials; i++)
    {
        update_pf(i);
        update_vis(i);
        
        for(int t=0; t<num_steps; t++)
        {
            update_tan(t);
            update_msn(t);
            update_motor(t);
            update_response(t);
            
            if(response != -1)
            {
                response_time = t;
                
                for(int tt=response_time; tt<num_steps; tt++)
                {
                    update_tan(tt);
                    update_msn(tt);
                    update_motor(tt);
                }
                break;
            }
        }
        
        update_feedback_mixed_7525(i);
        update_dopamine_corr(i);
        update_pf_tan(i);
        update_vis_msn(i);
        
        record_trial(i);
        reset_trial();
    }
}

void simulate_reacquisition()
{
    for(int i=num_acquisition_trials+num_extinction_trials; i<num_trials; i++)
    {
        update_pf(i);
        update_vis(i);
        
        for(int t=0; t<num_steps; t++)
        {
            update_tan(t);
            update_msn(t);
            update_motor(t);
            update_response(t);
            
            if(response != -1)
            {
                response_time = t;
                
                for(int tt=response_time; tt<num_steps; tt++)
                {
                    update_tan(tt);
                    update_msn(tt);
                    update_motor(tt);
                }
                break;
            }
        }
        
        update_feedback_contingent(i);
        update_dopamine_corr(i);
        update_pf_tan(i);
        update_vis_msn(i);
        
        record_trial(i);
        reset_trial();
    }
}

void simulate_nc_25()
{
    for(int i=0; i<num_simulations; i++)
    {
        printf("simulating iteration - %i\n", i);
        current_context = 1;
        simulate_acquisition();
        simulate_intervention_nc_25();
        simulate_reacquisition();
        record_simulation(i);
        reset_simulation();
        shuffle_stimuli();
    }
    
    compute_average_results();
    printf("writing nc_25 files\n");
    write_data_file(0);
    reset_averages();
}

void simulate_nc_40()
{
    for(int i=0; i<num_simulations; i++)
    {
        printf("simulating iteration - %i\n", i);
        current_context = 1;
        simulate_acquisition();
        simulate_intervention_nc_40();
        simulate_reacquisition();
        record_simulation(i);
        reset_simulation();
        shuffle_stimuli();
    }
    
    compute_average_results();
    printf("writing nc_40 files\n");
    write_data_file(1);
    reset_averages();
}

void simulate_nc_63()
{
    
    for(int i=0; i<num_simulations; i++)
    {
        printf("simulating iteration - %i\n", i);
        current_context = 1;
        simulate_acquisition();
        simulate_intervention_nc_63();
        simulate_reacquisition();
        record_simulation(i);
        reset_simulation();
        shuffle_stimuli();
    }
    
    compute_average_results();
    printf("writing nc_63 files\n");
    write_data_file(2);
    reset_averages();
}

void simulate_mixed_7525()
{
    for(int i=0; i<num_simulations; i++)
    {
        printf("simulating iteration - %i\n", i);
        current_context = 1;
        simulate_acquisition();
        simulate_intervention_mixed_7525();
        simulate_reacquisition();
        record_simulation(i);
        reset_simulation();
        shuffle_stimuli();
    }
    
    compute_average_results();
    printf("writing mixed_7525 files\n");
    write_data_file(3);
    reset_averages();
}

void set_params()
{
    num_acquisition_trials = 300;
    num_extinction_trials = 300;
    num_reacquisition_trials = 300;
    num_trials = num_acquisition_trials+num_extinction_trials+num_reacquisition_trials;
    num_simulations = 10;
	T = 3000;
	tau = 1;
	num_steps = T/tau;
    
    pf_amp = 1.65;
    num_pf_cells_per_context = 5;
    num_contexts = 3;
    num_pf_cells = num_contexts*num_pf_cells_per_context;
    num_pf_overlap = 1;
    current_context = -1;
    
    stim_onset = 1000;
    stim_duration = 1000;
    
    dim = 200;
    vis_amp = 25.0;
    vis_width = 50.0;
    vis_dist_x = 0.0;
    vis_dist_y = 0.0;
    
    w_noise_tan_L = 0.2;
    w_noise_tan_U = 0.2;
    w_noise_msn_L = 0.45;
    w_noise_msn_U = 0.55;
    
    lateral_inhibition_msn = 100.0;
    lateral_inhibition = 0.0;
    
    noise_tan_mu = 0.0;
    noise_tan_sigma = 1.0;
    
    noise_msn_mu = 200.0;
    noise_msn_sigma = 10.0;
    
    noise_motor_mu = 70.0;
    noise_motor_sigma = 1.0;
    
    noise = 0.0;
    
    w_vis_msn = 1.0;
    w_msn_motor = 0.1;
    
    // TAN-MSN
    w_tan_msn = 1000.0;
    
    w_ltp_vis_msn = 0.075e-8;
    w_ltd_vis_msn_1 = 0.05e-8;
    w_ltd_vis_msn_2 = 0.05e-11;
    
    w_ltp_pf_tan = 1.5e-5;
    w_ltd_pf_tan_1 = 0.9e-5;
    w_ltd_pf_tan_2 = 0.5e-6;
    
    nmda = 500.0;
    ampa = 100.0;
    
    confidence = 0.0;
    response = -1;
    response_time = -1;
    max_output = 0.0;
    
    resp_thresh = 5.0;
    
    obtained_feedback = 0.0;
    predicted_feedback = 0.0;
    w_prediction_error = 0.05;
    da = 0.0;
    da_base = 0.2;
    da_alpha = 1.0;
    da_beta = 25.0;
    r_theta = 0.85;
    
    spike_a = 1.0;
	spike_b = 100;
	spike_length = floor(7.64*spike_b);
	pause_decay = 0.0018;
	pause_mod_amp = 2.7;
}

void allocate_buffers()
{
    r_p_pos = (float *) calloc(num_trials, sizeof(float));
    r_p_neg = (float *) calloc(num_trials, sizeof(float));
    r_I_pos = (float *) calloc(num_trials, sizeof(float));
    r_I_neg = (float *) calloc(num_trials, sizeof(float));
    r_omega_pos = (float *) calloc(num_trials, sizeof(float));
    r_omega_neg = (float *) calloc(num_trials, sizeof(float));
    r_p_pos_mean = (float *) calloc(num_trials, sizeof(float));
    r_p_neg_mean = (float *) calloc(num_trials, sizeof(float));
    
    pf = (float **) calloc(num_contexts, sizeof(float*));
    pf_sum = (float **) calloc(num_contexts, sizeof(float*));
    w_pf_tan = (float **) calloc(num_contexts, sizeof(float*));
    for(int i=0; i<num_contexts; i++)
    {
        pf[i] = (float *) calloc(num_pf_cells_per_context, sizeof(float));
        pf_sum[i] = (float *) calloc(num_pf_cells_per_context, sizeof(float));
        w_pf_tan[i] = (float *) calloc(num_pf_cells_per_context, sizeof(float));
    }
    
    vis = (float *) calloc(dim*dim, sizeof(float));
    vis_sum = (float *) calloc(dim*dim, sizeof(float));
    w_vis_msn_A = (float *) calloc(dim*dim, sizeof(float));
    w_vis_msn_B = (float *) calloc(dim*dim, sizeof(float));
    w_vis_msn_C = (float *) calloc(dim*dim, sizeof(float));
    w_vis_msn_D = (float *) calloc(dim*dim, sizeof(float));
    vis_msn_act_A = (float *) calloc(num_steps, sizeof(float));
    vis_msn_act_B = (float *) calloc(num_steps, sizeof(float));
    vis_msn_act_C = (float *) calloc(num_steps, sizeof(float));
    vis_msn_act_D = (float *) calloc(num_steps, sizeof(float));
    
    spike = (float*) calloc(spike_length, sizeof(float));
    
    stim = (float **) calloc(3, sizeof(float *));
    stim[0] = (float *) calloc(num_trials, sizeof(float));
    stim[1] = (float *) calloc(num_trials, sizeof(float));
    stim[2] = (float *) calloc(num_trials, sizeof(float));
    
    pf_mod = (float *) calloc(num_steps, sizeof(float));
    pause_mod = (float *) calloc(num_steps, sizeof(float));
    pf_tan_act = (float *) calloc(num_steps, sizeof(float));
    
    tan_v = (float *) calloc(num_steps, sizeof(float));
    tan_u = (float *) calloc(num_steps, sizeof(float));
    tan_spikes = (int *) calloc(num_steps, sizeof(float));
    tan_output = (float *) calloc(num_steps, sizeof(float));
    
    msn_v_A = (float *) calloc(num_steps, sizeof(float));
    msn_u_A = (float *) calloc(num_steps, sizeof(float));
    msn_spikes_A = (int *) calloc(num_steps, sizeof(float));
    msn_output_A = (float *) calloc(num_steps, sizeof(float));
    
    msn_v_B = (float *) calloc(num_steps, sizeof(float));
    msn_u_B = (float *) calloc(num_steps, sizeof(float));
    msn_spikes_B = (int *) calloc(num_steps, sizeof(float));
    msn_output_B = (float *) calloc(num_steps, sizeof(float));
    
    msn_v_C = (float *) calloc(num_steps, sizeof(float));
    msn_u_C = (float *) calloc(num_steps, sizeof(float));
    msn_spikes_C = (int *) calloc(num_steps, sizeof(float));
    msn_output_C = (float *) calloc(num_steps, sizeof(float));
    
    msn_v_D = (float *) calloc(num_steps, sizeof(float));
    msn_u_D = (float *) calloc(num_steps, sizeof(float));
    msn_spikes_D = (int *) calloc(num_steps, sizeof(float));
    msn_output_D = (float *) calloc(num_steps, sizeof(float));
    
    motor_v_A = (float *) calloc(num_steps, sizeof(float));
    motor_spikes_A = (int *) calloc(num_steps, sizeof(float));
    motor_output_A = (float *) calloc(num_steps, sizeof(float));
    
    motor_v_B = (float *) calloc(num_steps, sizeof(float));
    motor_spikes_B = (int *) calloc(num_steps, sizeof(float));
    motor_output_B = (float *) calloc(num_steps, sizeof(float));
    
    motor_v_C = (float *) calloc(num_steps, sizeof(float));
    motor_spikes_C = (int *) calloc(num_steps, sizeof(float));
    motor_output_C = (float *) calloc(num_steps, sizeof(float));
    
    motor_v_D = (float *) calloc(num_steps, sizeof(float));
    motor_spikes_D = (int *) calloc(num_steps, sizeof(float));
    motor_output_D = (float *) calloc(num_steps, sizeof(float));
    
    outputs = (float *) calloc(4, sizeof(float));
    
    tan_v_record = (float **) calloc(num_trials, sizeof(float*));
    tan_output_record = (float **) calloc(num_trials, sizeof(float*));
    msn_v_A_record = (float **) calloc(num_trials, sizeof(float*));
    msn_v_B_record = (float **) calloc(num_trials, sizeof(float*));
    msn_v_C_record = (float **) calloc(num_trials, sizeof(float*));
    msn_v_D_record = (float **) calloc(num_trials, sizeof(float*));
    msn_output_A_record = (float **) calloc(num_trials, sizeof(float*));
    msn_output_B_record = (float **) calloc(num_trials, sizeof(float*));
    msn_output_C_record = (float **) calloc(num_trials, sizeof(float*));
    msn_output_D_record = (float **) calloc(num_trials, sizeof(float*));
    motor_v_A_record = (float **) calloc(num_trials, sizeof(float*));
    motor_v_B_record = (float **) calloc(num_trials, sizeof(float*));
    motor_v_C_record = (float **) calloc(num_trials, sizeof(float*));
    motor_v_D_record = (float **) calloc(num_trials, sizeof(float*));
    motor_output_A_record = (float **) calloc(num_trials, sizeof(float*));
    motor_output_B_record = (float **) calloc(num_trials, sizeof(float*));
    motor_output_C_record = (float **) calloc(num_trials, sizeof(float*));
    motor_output_D_record = (float **) calloc(num_trials, sizeof(float*));
    
    vis_record = (float **) calloc(num_trials, sizeof(float*));
    pf_record = (float **) calloc(num_trials, sizeof(float*));
    w_pf_tan_record = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_A_record = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_B_record = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_C_record = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_D_record = (float **) calloc(num_trials, sizeof(float*));
    
    w_pf_tan_record_ave = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_A_record_ave = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_B_record_ave = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_C_record_ave = (float **) calloc(num_trials, sizeof(float*));
    w_vis_msn_D_record_ave = (float **) calloc(num_trials, sizeof(float*));
    
    for(int i=0;i<num_trials; i++)
    {
        pf_record[i] = (float *) calloc(num_pf_cells, sizeof(float));
        w_pf_tan_record[i] = (float *) calloc(num_pf_cells, sizeof(float));
        
        vis_record[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_A_record[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_B_record[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_C_record[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_D_record[i] = (float *) calloc(dim*dim, sizeof(float));
        
        w_pf_tan_record_ave[i] = (float *) calloc(num_pf_cells, sizeof(float));
        w_vis_msn_A_record_ave[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_B_record_ave[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_C_record_ave[i] = (float *) calloc(dim*dim, sizeof(float));
        w_vis_msn_D_record_ave[i] = (float *) calloc(dim*dim, sizeof(float));
        
        tan_v_record[i] = (float *) calloc(num_steps, sizeof(float));
        tan_output_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_v_A_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_v_B_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_v_C_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_v_D_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_output_A_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_output_B_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_output_C_record[i] = (float *) calloc(num_steps, sizeof(float));
        msn_output_D_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_v_A_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_v_B_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_v_C_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_v_D_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_output_A_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_output_B_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_output_C_record[i] = (float *) calloc(num_steps, sizeof(float));
        motor_output_D_record[i] = (float *) calloc(num_steps, sizeof(float));
    }
    
    response_record = (float *) calloc(num_trials, sizeof(float));
    response_time_record = (float *) calloc(num_trials, sizeof(float));
    predicted_feedback_record = (float *) calloc(num_trials, sizeof(float));
    dopamine_record = (float *) calloc(num_trials, sizeof(float));
    correlation_record = (float *) calloc(num_trials, sizeof(float));
    confidence_record = (float *) calloc(num_trials, sizeof(float));
    accuracy_record = (float *) calloc(num_trials, sizeof(float));
    
    response_record_ave = (float *) calloc(num_trials, sizeof(float));
    response_time_record_ave = (float *) calloc(num_trials, sizeof(float));
    predicted_feedback_record_ave = (float *) calloc(num_trials, sizeof(float));
    dopamine_record_ave = (float *) calloc(num_trials, sizeof(float));
    correlation_record_ave = (float *) calloc(num_trials, sizeof(float));
    confidence_record_ave = (float *) calloc(num_trials, sizeof(float));
    accuracy_record_ave = (float *) calloc(num_trials, sizeof(float));
}

void load_stim()
{
    FILE *stim_fp = fopen("/Users/crossley/Documents/projects/research/renewal/renewal_spiking/input/stimuli_4_cat_200.txt", "r");
    for(int i=0; i<num_trials; i++)
    {
        fscanf(stim_fp, "%f %f %f\n", &stim[0][i], &stim[1][i], &stim[2][i]);
    }
    fclose(stim_fp);
}

void init_weights()
{
    // Init pf-tan weights
    for(int i=0; i<num_contexts; i++)
    {
        for(int j=0; j<num_pf_cells_per_context; j++)
        {
            w_pf_tan[i][j] = w_noise_tan_L + (w_noise_tan_U-w_noise_tan_L)*rand()/(float)RAND_MAX;
        }
    }
    
    // Init cortico-striatal weights
    for(int i=0; i<dim*dim; i++)
    {
        w_vis_msn_A[i] = w_noise_msn_L + (w_noise_msn_U-w_noise_msn_L)*rand()/(float)RAND_MAX;
        w_vis_msn_B[i] = w_noise_msn_L + (w_noise_msn_U-w_noise_msn_L)*rand()/(float)RAND_MAX;
        w_vis_msn_C[i] = w_noise_msn_L + (w_noise_msn_U-w_noise_msn_L)*rand()/(float)RAND_MAX;
        w_vis_msn_D[i] = w_noise_msn_L + (w_noise_msn_U-w_noise_msn_L)*rand()/(float)RAND_MAX;
    }
}

void init_buffers()
{
    for(int i=0; i<spike_length; i++)
	{
		spike[i] = spike_a*((float)i/spike_b)*exp(-1.0*(i-spike_b)/spike_b);
	}
}

void update_pf(int trial)
{
	// We assume that there are num_pf_overlap features common to all three contexts
	switch (current_context) {
		case 1: // Model is in context A
			
			// Turn on every context A Pf cell
			for(int i=0; i<num_pf_cells_per_context; i++)
			{
				pf[0][i] = 1.0;
			}
			
			// Turn on context B Pf cells that overlap with context A
			for(int i=0; i<num_pf_overlap; i++)
			{
				pf[1][i] = 1.0;
			}
			
			// Turn on context C Pf cells that overlap with context A
			for(int i=0; i<num_pf_overlap; i++)
			{
				pf[2][i] = 1.0;
			}
			
			break;
            
		case 2: // Model is in context B
			
			for(int i=0; i<num_pf_cells_per_context; i++)
			{
				pf[1][i] = 1.0;
			}
			
			for(int i=0; i<num_pf_overlap; i++)
			{
				pf[2][i] = 1.0;
			}
			
			for(int i=0; i<num_pf_overlap; i++)
			{
				pf[0][i] = 1.0;
			}
			
			break;
			
		case 3: // Model is in context C
			
			for(int i=0; i<num_pf_cells_per_context; i++)
			{
				pf[2][i] = 1.0;
			}
			
			for(int i=0; i<num_pf_overlap; i++)
			{
				pf[0][i] = 1.0;
			}
			
			for(int i=0; i<num_pf_overlap; i++)
			{
				pf[1][i] = 1.0;
			}
			
			break;
			
		default:
			break;
	}
	
	// Pf_mod is a decaying exponential with initial height = pf_amp*(num_pf_cells_per_context+num_pf_overlap)
	memset(pf_mod,0,stim_onset*sizeof(float));
	for(int i=0; i<num_steps; i++)
	{
		pf_mod[i] = pf_amp*(num_pf_cells_per_context+num_pf_overlap)*exp(-pause_decay*i);
	}
	
	// pause_mod is a square wave with a decaying expoential tail, also with initial height = pf_amp*(num_pf_cells_per_context+num_pf_overlap)
	memset(pause_mod,0,stim_onset*sizeof(float));
	for(int i=stim_onset; i<stim_onset+stim_duration; i++)
	{
		pause_mod[i] = pf_amp*(num_pf_cells_per_context+num_pf_overlap);
	}
	
	for(int i=stim_onset+stim_duration; i<num_steps; i++)
	{
		pause_mod[i] = pf_mod[i-(stim_onset+stim_duration)];
	}
    
	// Now, we figure out the collective pf-tan synaptic strength to be used in the TAN modulatory equation
	// This should basically be some kind of weighted average so that the Pf Cells in the active context
	// influence it most. With just two contexts, the old version of the model used:
	// Pf_TAN = (Pf_TAN_A*Pf_amp_A+Pf_TAN_B*Pf_amp_B)/(Pf_amp_A+Pf_amp_B);
    
	pf_tan_mod = 0.0;
    for(int i=0; i<num_contexts; i++)
    {
        for(int j=0; j<num_pf_cells_per_context; j++)
        {
			pf_tan_mod += pf[i][j]*w_pf_tan[i][j]*pf_amp;
		}
	}
	
	pf_tan_mod = pf_tan_mod / (float) pf_amp*(num_pf_cells_per_context+(num_contexts-1.0)*num_pf_overlap);
	
    memset(pf_tan_act,0,num_steps*sizeof(float));
    for(int i=0; i<num_contexts; i++)
	{
		for(int j=0; j<num_pf_cells_per_context; j++)
		{
            for(int k=stim_onset; k<stim_onset+stim_duration; k++)
            {
                pf_tan_act[k] +=  w_pf_tan[i][j]*pf[i][j];
            }
		}
	}
}

void update_vis(int trial)
{
    vis_dist_x = 0.0;
    vis_dist_y = 0.0;
    
    for(int i=0; i<dim; i++)
    {
        for(int j=0; j<dim; j++)
        {
            vis_dist_x = stim[1][trial] - j;
            vis_dist_y = stim[2][trial] - i;
            
			vis[j + i*dim] = vis_amp*exp(-(vis_dist_x*vis_dist_x+vis_dist_y*vis_dist_y)/vis_width);
        }
    }
    
    vDSP_dotpr(vis, 1, w_vis_msn_A, 1, &vis_act_A, dim*dim);
    vDSP_dotpr(vis, 1, w_vis_msn_B, 1, &vis_act_B, dim*dim);
    vDSP_dotpr(vis, 1, w_vis_msn_C, 1, &vis_act_C, dim*dim);
    vDSP_dotpr(vis, 1, w_vis_msn_D, 1, &vis_act_D, dim*dim);
    
    for(int k=stim_onset; k<stim_onset+stim_duration; k++)
    {
        vis_msn_act_A[k] = vis_act_A;
        vis_msn_act_B[k] = vis_act_B;
        vis_msn_act_C[k] = vis_act_C;
        vis_msn_act_D[k] = vis_act_D;
    }
}

void update_tan(int step)
{
	// IB -- intrinsically bursting
	float C=100.0, vr=-75.0, vt=-45.0, k=1.2;
	float a=0.01, b=5.0, c=-56.0, d=130.0;
	float vpeak=60.0;
	float E=950.0;
    
    int i = step;
    tan_v[0] = vr;
    
    noise = randn(noise_tan_mu, noise_tan_sigma);
    tan_v[i+1]=tan_v[i]+tau*(k*(tan_v[i]-vr)*(tan_v[i]-vt)-tan_u[i]+E+pf_tan_act[i]+noise)/C;
    tan_u[i+1]=tan_u[i]+tau*a*(b*(tan_v[i]-vr)-tan_u[i]+pf_tan_mod*pause_mod_amp*pause_mod[i]);
    
    if(tan_v[i+1]>=vpeak)
    {
        tan_v[i]=vpeak;
        tan_v[i+1]=c;
        tan_u[i+1]=tan_u[i+1]+d;
        tan_spikes[i+1]=1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                tan_output[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                tan_output[i+j] += spike[j];
            }
        }
        
    } else
    {
        tan_spikes[i+1]=0;
    }
}

void update_msn(int step)
{
    // Izichevich msn
	float C=50, vr=-80, vt=-25, k=1;
	float a=0.01, b=-20, c=-55, d=150;
	float vpeak=40;
	
    int i = step;
	msn_v_A[0] = vr;
    msn_v_B[0] = vr;
    msn_v_C[0] = vr;
    msn_v_D[0] = vr;
    
    noise = randn(noise_msn_mu, noise_msn_sigma);
    lateral_inhibition = lateral_inhibition_msn*(msn_output_B[i]+msn_output_C[i]+msn_output_D[i]);
    msn_v_A[i+1] = msn_v_A[i] + tau*(k*(msn_v_A[i]-vr)*(msn_v_A[i]-vt)-msn_u_A[i]+ w_vis_msn*pos(vis_msn_act_A[i]-w_tan_msn*tan_output[i])-lateral_inhibition+noise)/C;
    msn_u_A[i+1] = msn_u_A[i]+tau*a*(b*(msn_v_A[i]-vr)-msn_u_A[i]);
    
    noise = randn(noise_msn_mu, noise_msn_sigma);
    lateral_inhibition = lateral_inhibition_msn*(msn_output_A[i]+msn_output_C[i]+msn_output_D[i]);
    msn_v_B[i+1] = msn_v_B[i] + tau*(k*(msn_v_B[i]-vr)*(msn_v_B[i]-vt)-msn_u_B[i]+ w_vis_msn*pos(vis_msn_act_B[i]-w_tan_msn*tan_output[i])-lateral_inhibition+noise)/C;
    msn_u_B[i+1] = msn_u_B[i]+tau*a*(b*(msn_v_B[i]-vr)-msn_u_B[i]);
    
    noise = randn(noise_msn_mu, noise_msn_sigma);
    lateral_inhibition = lateral_inhibition_msn*(msn_output_A[i]+msn_output_B[i]+msn_output_D[i]);
    msn_v_C[i+1] = msn_v_C[i] + tau*(k*(msn_v_C[i]-vr)*(msn_v_C[i]-vt)-msn_u_C[i]+ w_vis_msn*pos(vis_msn_act_C[i]-w_tan_msn*tan_output[i])-lateral_inhibition+noise)/C;
    msn_u_C[i+1] = msn_u_C[i]+tau*a*(b*(msn_v_C[i]-vr)-msn_u_C[i]);
    
    noise = randn(noise_msn_mu, noise_msn_sigma);
    lateral_inhibition = lateral_inhibition_msn*(msn_output_A[i]+msn_output_B[i]+msn_output_C[i]);
    msn_v_D[i+1] = msn_v_D[i] + tau*(k*(msn_v_D[i]-vr)*(msn_v_D[i]-vt)-msn_u_D[i]+ w_vis_msn*pos(vis_msn_act_D[i]-w_tan_msn*tan_output[i])-lateral_inhibition+noise)/C;
    msn_u_D[i+1] = msn_u_D[i]+tau*a*(b*(msn_v_D[i]-vr)-msn_u_D[i]);
    
    if(msn_v_A[i+1]>=vpeak)
    {
        msn_v_A[i]=vpeak;
        msn_v_A[i+1]=c;
        msn_u_A[i+1]=msn_u_A[i+1]+d;
        msn_spikes_A[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                msn_output_A[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                msn_output_A[i+j] += spike[j];
            }
        }
        
    } else
    {
        msn_spikes_A[i+1] = 0;
    }
    
    if(msn_v_B[i+1]>=vpeak)
    {
        msn_v_B[i]=vpeak;
        msn_v_B[i+1]=c;
        msn_u_B[i+1]=msn_u_B[i+1]+d;
        msn_spikes_B[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                msn_output_B[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                msn_output_B[i+j] += spike[j];
            }
        }
        
    } else
    {
        msn_spikes_B[i+1] = 0;
    }
    
    if(msn_v_C[i+1]>=vpeak)
    {
        msn_v_C[i]=vpeak;
        msn_v_C[i+1]=c;
        msn_u_C[i+1]=msn_u_C[i+1]+d;
        msn_spikes_C[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                msn_output_C[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                msn_output_C[i+j] += spike[j];
            }
        }
        
    } else
    {
        msn_spikes_C[i+1] = 0;
    }
    
    if(msn_v_D[i+1]>=vpeak)
    {
        msn_v_D[i]=vpeak;
        msn_v_D[i+1]=c;
        msn_u_D[i+1]=msn_u_D[i+1]+d;
        msn_spikes_D[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                msn_output_D[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                msn_output_D[i+j] += spike[j];
            }
        }
        
    } else
    {
        msn_spikes_D[i+1] = 0;
    }
}

void update_motor(int step)
{
	// motor
	float C=25, vr=-60, vt=-40, k=0.7;
	float c=-50;
	float vpeak=35;
	
    int i = step;
	motor_v_A[0] = vr;
    motor_v_B[0] = vr;
    motor_v_C[0] = vr;
    motor_v_D[0] = vr;
    
    noise = randn(noise_motor_mu, noise_motor_sigma);
    motor_v_A[i+1] = motor_v_A[i] + tau*(k*(motor_v_A[i]-vr)*(motor_v_A[i]-vt)+w_msn_motor*msn_output_A[i]+noise)/C;
    
    noise = randn(noise_motor_mu, noise_motor_sigma);
    motor_v_B[i+1] = motor_v_B[i] + tau*(k*(motor_v_B[i]-vr)*(motor_v_B[i]-vt)+w_msn_motor*msn_output_B[i]+noise)/C;
    
    noise = randn(noise_motor_mu, noise_motor_sigma);
    motor_v_C[i+1] = motor_v_C[i] + tau*(k*(motor_v_C[i]-vr)*(motor_v_C[i]-vt)+w_msn_motor*msn_output_C[i]+noise)/C;
    
    noise = randn(noise_motor_mu, noise_motor_sigma);
    motor_v_D[i+1] = motor_v_D[i] + tau*(k*(motor_v_D[i]-vr)*(motor_v_D[i]-vt)+w_msn_motor*msn_output_D[i]+noise)/C;
    
    if(motor_v_A[i+1]>=vpeak)
    {
        motor_v_A[i]=vpeak;
        motor_v_A[i+1]=c;
        motor_spikes_A[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                motor_output_A[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                motor_output_A[i+j] += spike[j];
            }
        }
        
    } else
    {
        motor_spikes_A[i+1] = 0;
    }
    
    if(motor_v_B[i+1]>=vpeak)
    {
        motor_v_B[i]=vpeak;
        motor_v_B[i+1]=c;
        motor_spikes_B[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                motor_output_B[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                motor_output_B[i+j] += spike[j];
            }
        }
        
    } else
    {
        motor_spikes_B[i+1] = 0;
    }
    
    if(motor_v_C[i+1]>=vpeak)
    {
        motor_v_C[i]=vpeak;
        motor_v_C[i+1]=c;
        motor_spikes_C[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                motor_output_C[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                motor_output_C[i+j] += spike[j];
            }
        }
        
    } else
    {
        motor_spikes_C[i+1] = 0;
    }
    
    if(motor_v_D[i+1]>=vpeak)
    {
        motor_v_D[i]=vpeak;
        motor_v_D[i+1]=c;
        motor_spikes_D[i+1] = 1;
        
        if(i<num_steps-spike_length)
        {
            for(int j=0; j<spike_length; j++)
            {
                motor_output_D[i+j] += spike[j];
            }
        } else
        {
            for(int j=0; j<num_steps-i; j++)
            {
                motor_output_D[i+j] += spike[j];
            }
        }
        
    } else
    {
        motor_spikes_D[i+1] = 0;
    }
}

void update_response(int step)
{
    outputs[0] = motor_output_A[step];
    outputs[1] = motor_output_B[step];
    outputs[2] = motor_output_C[step];
    outputs[3] = motor_output_D[step];
    
    vDSP_maxmgvi(outputs, 1, &max_output, (unsigned long *) &response, 4);
    vDSP_vsort(outputs, 4, -1);
    
    if(outputs[0] > resp_thresh)
    {
        response++;
        confidence = (outputs[0]-outputs[1])/(float)outputs[0];
    } else
    {
        response = -1;
        confidence = -1;
    }
}

void update_feedback_contingent(int trial)
{
    if(response == -1)
    {
        outputs[0] = motor_output_A[num_steps-1];
        outputs[1] = motor_output_B[num_steps-1];
        outputs[2] = motor_output_C[num_steps-1];
        outputs[3] = motor_output_D[num_steps-1];
        
        vDSP_maxmgvi(outputs, 1, &max_output, (unsigned long *) &response, 4);
        vDSP_vsort(outputs, 4, -1);
        
        response++;
        response_time = num_steps;
        
        confidence = (outputs[0]-outputs[1])/(float)outputs[0];
    }
    
    // Give contingent feedback to both correct and incorrect responses
    obtained_feedback = stim[0][trial] == (float) response ? 1 : -1;
}

void update_feedback_nc_25(int trial)
{
    if(response == -1)
    {
        outputs[0] = motor_output_A[num_steps-1];
        outputs[1] = motor_output_B[num_steps-1];
        outputs[2] = motor_output_C[num_steps-1];
        outputs[3] = motor_output_D[num_steps-1];
        
        vDSP_maxmgvi(outputs, 1, &max_output, (unsigned long *) &response, 4);
        vDSP_vsort(outputs, 4, -1);
        
        response++;
        response_time = num_steps;
        
        confidence = (outputs[0]-outputs[1])/(float)outputs[0];
    }
    
    // Give random feedback
    obtained_feedback = rand()/(float)RAND_MAX < 0.25 ? 1 : -1;
}

void update_feedback_nc_40(int trial)
{
    if(response == -1)
    {
        outputs[0] = motor_output_A[num_steps-1];
        outputs[1] = motor_output_B[num_steps-1];
        outputs[2] = motor_output_C[num_steps-1];
        outputs[3] = motor_output_D[num_steps-1];
        
        vDSP_maxmgvi(outputs, 1, &max_output, (unsigned long *) &response, 4);
        vDSP_vsort(outputs, 4, -1);
        
        response++;
        response_time = num_steps;
        
        confidence = (outputs[0]-outputs[1])/(float)outputs[0];
    }
    
    // Give random feedback
    obtained_feedback = rand()/(float)RAND_MAX < 0.40 ? 1 : -1;
}

void update_feedback_nc_63(int trial)
{
    if(response == -1)
    {
        outputs[0] = motor_output_A[num_steps-1];
        outputs[1] = motor_output_B[num_steps-1];
        outputs[2] = motor_output_C[num_steps-1];
        outputs[3] = motor_output_D[num_steps-1];
        
        vDSP_maxmgvi(outputs, 1, &max_output, (unsigned long *) &response, 4);
        vDSP_vsort(outputs, 4, -1);
        
        response++;
        response_time = num_steps;
        
        confidence = (outputs[0]-outputs[1])/(float)outputs[0];
    }
    
    // Give random feedback
    obtained_feedback = rand()/(float)RAND_MAX < 0.63 ? 1 : -1;
}

void update_feedback_mixed_7525(int trial)
{
    if(response == -1)
    {
        outputs[0] = motor_output_A[num_steps-1];
        outputs[1] = motor_output_B[num_steps-1];
        outputs[2] = motor_output_C[num_steps-1];
        outputs[3] = motor_output_D[num_steps-1];
        
        vDSP_maxmgvi(outputs, 1, &max_output, (unsigned long *) &response, 4);
        vDSP_vsort(outputs, 4, -1);
        
        response++;
        response_time = num_steps;
        
        confidence = (outputs[0]-outputs[1])/(float)outputs[0];
    }
    
    if(rand()/(float)RAND_MAX > 0.25)
    {
        // Give random nc_25 feedback
        obtained_feedback = rand()/(float)RAND_MAX > 0.75 ? 1 : -1;
    }else{
        // Give contingent feedback to both correct and incorrect responses
        obtained_feedback = stim[0][trial] == (float) response ? 1 : -1;
    }
}

void update_dopamine_rpe(int trial)
{
    predicted_feedback += w_prediction_error*(obtained_feedback-predicted_feedback);
    da = cap(0.8*(obtained_feedback-predicted_feedback) + 0.2);
}

void update_dopamine_corr(int trial)
{
    predicted_feedback += w_prediction_error*(obtained_feedback-predicted_feedback);
    
    // update dopamine via interative correlation method
    if(obtained_feedback == 1.0)
    {
        r_p_pos[trial] = confidence;
        r_p_neg[trial] = 0.0;
        
        r_I_pos[trial] = 1.0;
        r_I_neg[trial] = 0.0;
        
    }else if(obtained_feedback == -1.0)
    {
        r_p_pos[trial] = 0.0;
        r_p_neg[trial] = confidence;
        
        r_I_pos[trial] = 0.0;
        r_I_neg[trial] = 1.0;
    }
    
    for(int i=0; i<=trial; i++)
    {
        r_omega_pos[trial] += pow(r_theta, trial-i) * r_I_pos[i];
        r_omega_neg[trial] += pow(r_theta, trial-i) * r_I_neg[i];
    }
    
    for(int i=0; i<=trial; i++)
    {
        r_p_pos_mean[trial] += (1.0/r_omega_pos[trial]) * pow(r_theta, trial-i) * r_p_pos[i];
        r_p_neg_mean[trial] += (1.0/r_omega_neg[trial]) * pow(r_theta, trial-i) * r_p_neg[i];
    }
    
    if(trial > 0)
    {
        if(isnan(r_p_pos_mean[trial-1]) || r_omega_pos[trial] == 0.0)
        {
            correlation = fabs(- r_p_neg[trial] / r_omega_neg[trial] - (r_omega_neg[trial] - 1.0) * r_p_neg_mean[trial-1] / r_omega_neg[trial]);
            
        }else if(isnan(r_p_neg_mean[trial-1]) || r_omega_neg[trial] == 0.0)
        {
            
            correlation = fabs(+ r_p_pos[trial] / r_omega_pos[trial] + (r_omega_pos[trial] - 1.0) * r_p_pos_mean[trial-1] / r_omega_pos[trial]);
            
        }else
        {
            correlation = fabs(
                               + r_p_pos[trial] / r_omega_pos[trial] + (r_omega_pos[trial] - 1.0) * r_p_pos_mean[trial-1] / r_omega_pos[trial]
                               - r_p_neg[trial] / r_omega_neg[trial] - (r_omega_neg[trial] - 1.0) * r_p_neg_mean[trial-1] / r_omega_neg[trial]);
        }
    }else
    {
        correlation = 0.1;
    }
    
    correlation = cap(correlation);
    
    if(trial > 25)
    {
        da = cap(da_alpha*correlation*((obtained_feedback)-(2.0*confidence-1.0)) + da_base*(1.0-exp(-da_beta*correlation)));
    }else
    {
        da = cap(da_alpha*0.2*((obtained_feedback)-(2.0*confidence-1.0)) + da_base*(1.0-exp(-da_beta*0.2)));
    }
}

void update_pf_tan(int trial)
{
	for(int i=0; i<num_contexts; i++)
	{
		for(int j=0; j<num_pf_cells_per_context; j++)
		{
            pf_sum[i][j] = 200*pos(pf[i][j]);
		}
	}
	
	for(int i=stim_onset; i<stim_onset+200; i++)
	{
		tan_sum += pos(tan_v[i]);
	}
    
	for(int i=0; i<num_contexts; i++)
	{
		for(int j=0; j<num_pf_cells_per_context; j++)
		{
			w_pf_tan[i][j] = cap( w_pf_tan[i][j]
                                 + w_ltp_pf_tan*pf_sum[i][j]*pos(tan_sum-20.0)*pos(da-da_base)*pos(1.0-w_pf_tan[i][j])
                                 - w_ltd_pf_tan_1*pf_sum[i][j]*pos(tan_sum-20.0)*pos(da_base-da)*pos(w_pf_tan[i][j])
                                 - w_ltd_pf_tan_2*pf_sum[i][j]*pos(20.0-tan_sum)*pos(tan_sum-10.0)*pos(w_pf_tan[i][j]));
            
            //            w_pf_tan[i][j] = trial > 300 ? 0.2 :  w_pf_tan[i][j];
		}
	}
}

void update_vis_msn(int trial)
{
    for(int i=0; i<dim*dim; i++)
    {
        vis_sum[i] = (float)stim_duration*vis[i];
    }
    
    for(int i=stim_onset; i<stim_onset+stim_duration; i++)
    {
        msn_sum_A += pos(msn_v_A[i]);
        msn_sum_B += pos(msn_v_B[i]);
        msn_sum_C += pos(msn_v_C[i]);
        msn_sum_D += pos(msn_v_D[i]);
    }
    
    //    printf("%f %f %f %f\n",msn_sum_A,msn_sum_B,msn_sum_C,msn_sum_D);
	
    for(int i=0; i<dim*dim; i++)
    {
        w_vis_msn_A[i] = cap(w_vis_msn_A[i]
                             + w_ltp_vis_msn*vis_sum[i]*pos(msn_sum_A-nmda)*pos(da-da_base)*pos(1.0-w_vis_msn_A[i])
                             - w_ltd_vis_msn_1*vis_sum[i]*pos(msn_sum_A-nmda)*pos(da_base-da)*pos(w_vis_msn_A[i])
                             - w_ltd_vis_msn_2*vis_sum[i]*pos(nmda-msn_sum_A)*pos(msn_sum_A-ampa)*pos(w_vis_msn_A[i]));
        
        w_vis_msn_B[i] = cap(w_vis_msn_B[i]
                             + w_ltp_vis_msn*vis_sum[i]*pos(msn_sum_B-nmda)*pos(da-da_base)*pos(1.0-w_vis_msn_B[i])
                             - w_ltd_vis_msn_1*vis_sum[i]*pos(msn_sum_B-nmda)*pos(da_base-da)*pos(w_vis_msn_B[i])
                             - w_ltd_vis_msn_2*vis_sum[i]*pos(nmda-msn_sum_B)*pos(msn_sum_B-ampa)*pos(w_vis_msn_B[i]));
        
        w_vis_msn_C[i] = cap(w_vis_msn_C[i]
                             + w_ltp_vis_msn*vis_sum[i]*pos(msn_sum_C-nmda)*pos(da-da_base)*pos(1.0-w_vis_msn_C[i])
                             - w_ltd_vis_msn_1*vis_sum[i]*pos(msn_sum_C-nmda)*pos(da_base-da)*pos(w_vis_msn_C[i])
                             - w_ltd_vis_msn_2*vis_sum[i]*pos(nmda-msn_sum_C)*pos(msn_sum_C-ampa)*pos(w_vis_msn_C[i]));
        
        w_vis_msn_D[i] = cap(w_vis_msn_D[i]
                             + w_ltp_vis_msn*vis_sum[i]*pos(msn_sum_D-nmda)*pos(da-da_base)*pos(1.0-w_vis_msn_D[i])
                             - w_ltd_vis_msn_1*vis_sum[i]*pos(msn_sum_D-nmda)*pos(da_base-da)*pos(w_vis_msn_D[i])
                             - w_ltd_vis_msn_2*vis_sum[i]*pos(nmda-msn_sum_D)*pos(msn_sum_D-ampa)*pos(w_vis_msn_D[i]));
    }
}

void record_trial(int trial)
{
    memcpy(vis_record[trial], vis, dim*dim*sizeof(float));
    
    for(int i=0; i<num_contexts; i++)
    {
        for(int j=0; j<num_pf_cells_per_context; j++)
        {
            pf_record[trial][j+i*num_pf_cells_per_context] = pf[i][j];
            w_pf_tan_record[trial][j+i*num_pf_cells_per_context] = w_pf_tan[i][j];
        }
    }
    
	memcpy(tan_v_record[trial], tan_v, num_steps*sizeof(float));
	memcpy(msn_v_A_record[trial], msn_v_A, num_steps*sizeof(float));
    memcpy(msn_v_B_record[trial], msn_v_B, num_steps*sizeof(float));
    memcpy(msn_v_C_record[trial], msn_v_C, num_steps*sizeof(float));
    memcpy(msn_v_D_record[trial], msn_v_D, num_steps*sizeof(float));
	memcpy(motor_v_A_record[trial], motor_v_A, num_steps*sizeof(float));
    memcpy(motor_v_B_record[trial], motor_v_B, num_steps*sizeof(float));
    memcpy(motor_v_C_record[trial], motor_v_C, num_steps*sizeof(float));
    memcpy(motor_v_D_record[trial], motor_v_D, num_steps*sizeof(float));
    
	memcpy(tan_output_record[trial], tan_output, num_steps*sizeof(float));
	memcpy(msn_output_A_record[trial], msn_output_A, num_steps*sizeof(float));
    memcpy(msn_output_B_record[trial], msn_output_B, num_steps*sizeof(float));
    memcpy(msn_output_C_record[trial], msn_output_C, num_steps*sizeof(float));
    memcpy(msn_output_D_record[trial], msn_output_D, num_steps*sizeof(float));
	memcpy(motor_output_A_record[trial], motor_output_A, num_steps*sizeof(float));
    memcpy(motor_output_B_record[trial], motor_output_B, num_steps*sizeof(float));
    memcpy(motor_output_C_record[trial], motor_output_C, num_steps*sizeof(float));
    memcpy(motor_output_D_record[trial], motor_output_D, num_steps*sizeof(float));
    
    memcpy(w_vis_msn_A_record[trial], w_vis_msn_A, dim*dim*sizeof(float));
    memcpy(w_vis_msn_B_record[trial], w_vis_msn_B, dim*dim*sizeof(float));
    memcpy(w_vis_msn_C_record[trial], w_vis_msn_C, dim*dim*sizeof(float));
    memcpy(w_vis_msn_D_record[trial], w_vis_msn_D, dim*dim*sizeof(float));
    
    response_record[trial] = response;
    response_time_record[trial] = response_time;
    predicted_feedback_record[trial] = predicted_feedback;
    dopamine_record[trial] = da;
    correlation_record[trial] = correlation;
    confidence_record[trial] = confidence;
    accuracy_record[trial] = stim[0][trial] == response ? 1 : 0;
}

void record_simulation(int simulation)
{
    for(int i=0; i<num_trials; i++)
    {
        response_record_ave[i] += response_record[i];
        response_time_record_ave[i] += response_time_record[i];
        accuracy_record_ave[i] += accuracy_record[i];
        dopamine_record_ave[i] += dopamine_record[i];
        predicted_feedback_record_ave[i] += predicted_feedback_record[i];
        correlation_record_ave[i] += correlation_record[i];
        confidence_record_ave[i] += confidence_record[i];
        
        vDSP_vadd(w_pf_tan_record_ave[i], 1, w_pf_tan_record[i], 1, w_pf_tan_record_ave[i], 1, num_pf_cells);
        vDSP_vadd(w_vis_msn_A_record_ave[i], 1, w_vis_msn_A_record[i], 1, w_vis_msn_A_record_ave[i], 1, dim*dim);
        vDSP_vadd(w_vis_msn_B_record_ave[i], 1, w_vis_msn_B_record[i], 1, w_vis_msn_B_record_ave[i], 1, dim*dim);
        vDSP_vadd(w_vis_msn_C_record_ave[i], 1, w_vis_msn_C_record[i], 1, w_vis_msn_C_record_ave[i], 1, dim*dim);
        vDSP_vadd(w_vis_msn_D_record_ave[i], 1, w_vis_msn_D_record[i], 1, w_vis_msn_D_record_ave[i], 1, dim*dim);
    }
}

void reset_trial()
{
    response = -1;
    response_time = -1;
    obtained_feedback = 0.0;
    confidence = 0.0;
    
    msn_sum_A = 0.0;
    msn_sum_B = 0.0;
    msn_sum_C = 0.0;
    msn_sum_D = 0.0;
    
    memset(vis, 0, dim*dim*sizeof(float));
    memset(vis_sum, 0, dim*dim*sizeof(float));
    
    for(int i=0; i<num_contexts; i++)
    {
        for(int j=0; j<num_pf_cells_per_context; j++)
        {
            pf[i][j] = 0.0;
            pf_sum[i][j] = 0.0;
        }
    }
    tan_sum = 0.0;
    
	memset(tan_v, 0, num_steps*sizeof(float));
	memset(msn_v_A, 0, num_steps*sizeof(float));
    memset(msn_v_B, 0, num_steps*sizeof(float));
    memset(msn_v_C, 0, num_steps*sizeof(float));
    memset(msn_v_D, 0, num_steps*sizeof(float));
	memset(motor_v_A, 0, num_steps*sizeof(float));
    memset(motor_v_B, 0, num_steps*sizeof(float));
    memset(motor_v_C, 0, num_steps*sizeof(float));
    memset(motor_v_D, 0, num_steps*sizeof(float));
    
	memset(tan_output, 0, num_steps*sizeof(float));
	memset(msn_output_A, 0, num_steps*sizeof(float));
    memset(msn_output_B, 0, num_steps*sizeof(float));
    memset(msn_output_C, 0, num_steps*sizeof(float));
    memset(msn_output_D, 0, num_steps*sizeof(float));
	memset(motor_output_A, 0, num_steps*sizeof(float));
    memset(motor_output_B, 0, num_steps*sizeof(float));
    memset(motor_output_C, 0, num_steps*sizeof(float));
    memset(motor_output_D, 0, num_steps*sizeof(float));
}

void reset_simulation()
{
    obtained_feedback = 0.0;
    predicted_feedback = 0.0;
    correlation = 0.0;
    
    memset(response_record, 0, num_trials*sizeof(float));
    memset(response_time_record, 0, num_trials*sizeof(float));
    memset(accuracy_record, 0, num_trials*sizeof(float));
    memset(dopamine_record, 0, num_trials*sizeof(float));
    memset(predicted_feedback_record, 0, num_trials*sizeof(float));
    memset(correlation_record, 0, num_trials*sizeof(float));
    memset(confidence_record, 0, num_trials*sizeof(float));
    
    memset(r_p_pos, 0, num_trials*sizeof(float));
    memset(r_p_neg, 0, num_trials*sizeof(float));
    memset(r_I_pos, 0, num_trials*sizeof(float));
    memset(r_I_neg, 0, num_trials*sizeof(float));
    memset(r_omega_pos, 0, num_trials*sizeof(float));
    memset(r_omega_neg, 0, num_trials*sizeof(float));
    memset(r_p_pos_mean, 0, num_trials*sizeof(float));
    memset(r_p_neg_mean, 0, num_trials*sizeof(float));
    
    for(int i=0; i<num_contexts; i++)
    {
        for(int j=0; j<num_pf_cells_per_context; j++)
        {
            w_pf_tan[i][j] = 0.0;
        }
    }
    
    memset(w_vis_msn_A, 0, dim*dim*sizeof(float));
    memset(w_vis_msn_B, 0, dim*dim*sizeof(float));
    memset(w_vis_msn_C, 0, dim*dim*sizeof(float));
    memset(w_vis_msn_D, 0, dim*dim*sizeof(float));
    
    init_weights();
}
void compute_average_results()
{
    for(int i=0; i<num_trials; i++)
    {
        response_record_ave[i] /= (float)num_simulations;
        response_time_record_ave[i] /= (float)num_simulations;
        accuracy_record_ave[i] /= (float)num_simulations;
        dopamine_record_ave[i] /= (float)num_simulations;
        predicted_feedback_record_ave[i] /= (float)num_simulations;
        correlation_record_ave[i] /= (float)num_simulations;
        confidence_record_ave[i] /= (float)num_simulations;
        
        vDSP_vsdiv(w_pf_tan_record_ave[i], 1, &num_simulations, w_pf_tan_record_ave[i], 1, num_pf_cells);
        vDSP_vsdiv(w_vis_msn_A_record_ave[i], 1, &num_simulations, w_vis_msn_A_record_ave[i], 1, dim*dim);
        vDSP_vsdiv(w_vis_msn_B_record_ave[i], 1, &num_simulations, w_vis_msn_B_record_ave[i], 1, dim*dim);
        vDSP_vsdiv(w_vis_msn_C_record_ave[i], 1, &num_simulations, w_vis_msn_C_record_ave[i], 1, dim*dim);
        vDSP_vsdiv(w_vis_msn_D_record_ave[i], 1, &num_simulations, w_vis_msn_D_record_ave[i], 1, dim*dim);
    }
}

void reset_averages()
{
    memset(response_record_ave, 0, num_trials*sizeof(float));
    memset(response_time_record_ave, 0, num_trials*sizeof(float));
    memset(accuracy_record_ave, 0, num_trials*sizeof(float));
    memset(dopamine_record_ave, 0, num_trials*sizeof(float));
    memset(predicted_feedback_record_ave, 0, num_trials*sizeof(float));
    memset(correlation_record_ave, 0, num_trials*sizeof(float));
    memset(confidence_record_ave, 0, num_trials*sizeof(float));
}

void shuffle_stimuli()
{
	int temp_0, temp_1, temp_2, i, j;
    
    for(i=0; i<num_trials-1; i++)
	{
		j = i + rand() / ((float)RAND_MAX / (num_trials-i) + 1);
        
		temp_0 = stim[0][j];
		temp_1 = stim[1][j];
		temp_2 = stim[2][j];
		
		stim[0][j] = stim[0][i];
		stim[1][j] = stim[1][i];
		stim[2][j] = stim[2][i];
		
		stim[0][i] = temp_0;
		stim[1][i] = temp_1;
		stim[2][i] = temp_2;
	}
}

void write_data_file(int path)
{
        
    switch (path)
    {
        case 0:
            
            response_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/response.txt", "w");
            response_time_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/response_time.txt", "w");
            acc_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/acc.txt", "w");
            predicted_feedback_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/predicted_feedback.txt", "w");
            dopamine_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/dopamine.txt", "w");
            correlation_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/correlation.txt", "w");
            confidence_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/confidence.txt", "w");
            
            vis_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/vis.txt","w");
            w_vis_msn_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/w_vis_msn_A.txt","w");
            w_vis_msn_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/w_vis_msn_B.txt","w");
            w_vis_msn_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/w_vis_msn_C.txt","w");
            w_vis_msn_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/w_vis_msn_D.txt","w");
            
            w_pf_tan_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/w_pf_tan.txt", "w");
            pf_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/pf.txt","w");
            
            tan_output_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/tan_output.txt","w");
            msn_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_output_A.txt","w");
            msn_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_output_B.txt","w");
            msn_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_output_C.txt","w");
            msn_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_output_D.txt","w");
            motor_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_output_A.txt","w");
            motor_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_output_B.txt","w");
            motor_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_output_C.txt","w");
            motor_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_output_D.txt","w");
            
            tan_v_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/tan_v.txt","w");
            msn_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_v_A.txt","w");
            msn_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_v_B.txt","w");
            msn_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_v_C.txt","w");
            msn_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/msn_v_D.txt","w");
            motor_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_v_A.txt","w");
            motor_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_v_B.txt","w");
            motor_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_v_C.txt","w");
            motor_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_25/mot_v_D.txt","w");
            break;
            
        case 1:
            
            response_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/response.txt", "w");
            response_time_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/response_time.txt", "w");
            acc_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/acc.txt", "w");
            predicted_feedback_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/predicted_feedback.txt", "w");
            dopamine_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/dopamine.txt", "w");
            correlation_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/correlation.txt", "w");
            confidence_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/confidence.txt", "w");
            
            vis_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/vis.txt","w");
            w_vis_msn_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/w_vis_msn_A.txt","w");
            w_vis_msn_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/w_vis_msn_B.txt","w");
            w_vis_msn_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/w_vis_msn_C.txt","w");
            w_vis_msn_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/w_vis_msn_D.txt","w");
            
            w_pf_tan_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/w_pf_tan.txt", "w");
            pf_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/pf.txt","w");
            
            tan_output_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/tan_output.txt","w");
            msn_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_output_A.txt","w");
            msn_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_output_B.txt","w");
            msn_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_output_C.txt","w");
            msn_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_output_D.txt","w");
            motor_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_output_A.txt","w");
            motor_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_output_B.txt","w");
            motor_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_output_C.txt","w");
            motor_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_output_D.txt","w");
            
            tan_v_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/tan_v.txt","w");
            msn_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_v_A.txt","w");
            msn_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_v_B.txt","w");
            msn_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_v_C.txt","w");
            msn_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/msn_v_D.txt","w");
            motor_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_v_A.txt","w");
            motor_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_v_B.txt","w");
            motor_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_v_C.txt","w");
            motor_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_40/mot_v_D.txt","w");
            break;
            
        case 2:
            
            response_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/response.txt", "w");
            response_time_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/response_time.txt", "w");
            acc_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/acc.txt", "w");
            predicted_feedback_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/predicted_feedback.txt", "w");
            dopamine_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/dopamine.txt", "w");
            correlation_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/correlation.txt", "w");
            confidence_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/confidence.txt", "w");
            
            vis_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/vis.txt","w");
            w_vis_msn_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/w_vis_msn_A.txt","w");
            w_vis_msn_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/w_vis_msn_B.txt","w");
            w_vis_msn_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/w_vis_msn_C.txt","w");
            w_vis_msn_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/w_vis_msn_D.txt","w");
            
            w_pf_tan_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/w_pf_tan.txt", "w");
            pf_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/pf.txt","w");
            
            tan_output_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/tan_output.txt","w");
            msn_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_output_A.txt","w");
            msn_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_output_B.txt","w");
            msn_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_output_C.txt","w");
            msn_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_output_D.txt","w");
            motor_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_output_A.txt","w");
            motor_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_output_B.txt","w");
            motor_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_output_C.txt","w");
            motor_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_output_D.txt","w");
            
            tan_v_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/tan_v.txt","w");
            msn_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_v_A.txt","w");
            msn_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_v_B.txt","w");
            msn_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_v_C.txt","w");
            msn_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/msn_v_D.txt","w");
            motor_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_v_A.txt","w");
            motor_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_v_B.txt","w");
            motor_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_v_C.txt","w");
            motor_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/nc_63/mot_v_D.txt","w");
            break;
            
        case 3:
            
            response_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/response.txt", "w");
            response_time_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/response_time.txt", "w");
            acc_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/acc.txt", "w");
            predicted_feedback_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/predicted_feedback.txt", "w");
            dopamine_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/dopamine.txt", "w");
            correlation_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/correlation.txt", "w");
            confidence_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/confidence.txt", "w");
            
            vis_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/vis.txt","w");
            w_vis_msn_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/w_vis_msn_A.txt","w");
            w_vis_msn_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/w_vis_msn_B.txt","w");
            w_vis_msn_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/w_vis_msn_C.txt","w");
            w_vis_msn_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/w_vis_msn_D.txt","w");
            
            w_pf_tan_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/w_pf_tan.txt", "w");
            pf_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/pf.txt","w");
            
            tan_output_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/tan_output.txt","w");
            msn_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_output_A.txt","w");
            msn_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_output_B.txt","w");
            msn_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_output_C.txt","w");
            msn_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_output_D.txt","w");
            motor_output_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_output_A.txt","w");
            motor_output_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_output_B.txt","w");
            motor_output_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_output_C.txt","w");
            motor_output_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_output_D.txt","w");
            
            tan_v_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/tan_v.txt","w");
            msn_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_v_A.txt","w");
            msn_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_v_B.txt","w");
            msn_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_v_C.txt","w");
            msn_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/msn_v_D.txt","w");
            motor_v_A_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_v_A.txt","w");
            motor_v_B_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_v_B.txt","w");
            motor_v_C_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_v_C.txt","w");
            motor_v_D_file = fopen("/Users/crossley/Documents/projects/research/unlearning_master/spiking/output/mixed_7525/mot_v_D.txt","w");
            break;
            
        default:
            break;
    }
    
    for(int i=0; i<num_trials; i++)
    {
        fprintf(response_file, "%f\n",response_record_ave[i]);
        fprintf(response_time_file, "%f\n",response_time_record_ave[i]);
        fprintf(acc_file, "%f\n",accuracy_record_ave[i]);
        fprintf(predicted_feedback_file, "%f\n", predicted_feedback_record_ave[i]);
        fprintf(dopamine_file, "%f\n", dopamine_record_ave[i]);
        fprintf(correlation_file, "%f\n", correlation_record_ave[i]);
        fprintf(confidence_file, "%f\n", confidence_record_ave[i]);
        
        for(int j=0; j<num_contexts; j++)
        {
            for(int k=0; k<num_pf_cells_per_context; k++)
            {
                fprintf(pf_file, "%f ", w_pf_tan_record[i][k+j*num_pf_cells_per_context]);
                fprintf(w_pf_tan_file, "%f ", w_pf_tan_record[i][k+j*num_pf_cells_per_context]);
            }
        }
        
        for(int j=0; j<dim*dim; j++)
        {
            fprintf(vis_file, "%f ", vis_record[i][j]);
            fprintf(w_vis_msn_A_file, "%f ", w_vis_msn_A_record_ave[i][j]);
            fprintf(w_vis_msn_B_file, "%f ", w_vis_msn_B_record_ave[i][j]);
            fprintf(w_vis_msn_C_file, "%f ", w_vis_msn_C_record_ave[i][j]);
            fprintf(w_vis_msn_D_file, "%f ", w_vis_msn_D_record_ave[i][j]);
        }
        
        for(int j=0; j<num_steps; j++)
        {
            fprintf(tan_output_file, "%f ", tan_output_record[i][j]);
            fprintf(msn_output_A_file, "%f ", msn_output_A_record[i][j]);
            fprintf(msn_output_B_file, "%f ", msn_output_B_record[i][j]);
            fprintf(msn_output_C_file, "%f ", msn_output_C_record[i][j]);
            fprintf(msn_output_D_file, "%f ", msn_output_D_record[i][j]);
            fprintf(motor_output_A_file, "%f ", motor_output_A_record[i][j]);
            fprintf(motor_output_B_file, "%f ", motor_output_B_record[i][j]);
            fprintf(motor_output_C_file, "%f ", motor_output_C_record[i][j]);
            fprintf(motor_output_D_file, "%f ", motor_output_D_record[i][j]);
            
            fprintf(tan_v_file, "%f ", tan_v_record[i][j]);
            fprintf(msn_v_A_file, "%f ", msn_v_A_record[i][j]);
            fprintf(msn_v_B_file, "%f ", msn_v_B_record[i][j]);
            fprintf(msn_v_C_file, "%f ", msn_v_C_record[i][j]);
            fprintf(msn_v_D_file, "%f ", msn_v_D_record[i][j]);
            fprintf(motor_v_A_file, "%f ", motor_v_A_record[i][j]);
            fprintf(motor_v_B_file, "%f ", motor_v_B_record[i][j]);
            fprintf(motor_v_C_file, "%f ", motor_v_C_record[i][j]);
            fprintf(motor_v_D_file, "%f ", motor_v_D_record[i][j]);
        }
        
        fprintf(vis_file, "\n");
        fprintf(w_pf_tan_file, "\n");
        fprintf(w_vis_msn_A_file, "\n");
        fprintf(w_vis_msn_B_file, "\n");
        fprintf(w_vis_msn_C_file, "\n");
        fprintf(w_vis_msn_D_file, "\n");
        fprintf(pf_file, "\n");
        fprintf(tan_output_file, "\n");
        fprintf(msn_output_A_file, "\n");
        fprintf(msn_output_B_file, "\n");
        fprintf(msn_output_C_file, "\n");
        fprintf(msn_output_D_file, "\n");
        fprintf(motor_output_A_file, "\n");
        fprintf(motor_output_B_file, "\n");
        fprintf(motor_output_C_file, "\n");
        fprintf(motor_output_D_file, "\n");
        fprintf(tan_v_file, "\n");
        fprintf(msn_v_A_file, "\n");
        fprintf(msn_v_B_file, "\n");
        fprintf(msn_v_C_file, "\n");
        fprintf(msn_v_D_file, "\n");
        fprintf(motor_v_A_file, "\n");
        fprintf(motor_v_B_file, "\n");
        fprintf(motor_v_C_file, "\n");
        fprintf(motor_v_D_file, "\n");
    }
    
    fclose(response_file);
    fclose(response_time_file);
    fclose(acc_file);
    fclose(predicted_feedback_file);
    fclose(dopamine_file);
    fclose(correlation_file);
    fclose(confidence_file);
    fclose(w_pf_tan_file);
    fclose(vis_file);
    fclose(w_vis_msn_A_file);
    fclose(w_vis_msn_B_file);
    fclose(w_vis_msn_C_file);
    fclose(w_vis_msn_D_file);
    fclose(pf_file);
    fclose(tan_output_file);
    fclose(msn_output_A_file);
    fclose(msn_output_B_file);
    fclose(msn_output_C_file);
    fclose(msn_output_D_file);
    fclose(motor_output_A_file);
    fclose(motor_output_B_file);
    fclose(motor_output_C_file);
    fclose(motor_output_D_file);
    fclose(tan_v_file);
    fclose(msn_v_A_file);
    fclose(msn_v_B_file);
    fclose(msn_v_C_file);
    fclose(msn_v_D_file);
    fclose(motor_v_A_file);
    fclose(motor_v_B_file);
    fclose(motor_v_C_file);
    fclose(motor_v_D_file);
}

float pos(float val)
{
    return val < 0 ? 0 : val;
}

float cap(float val)
{
    if(val < 0.0) val = 0.0;
    if(val > 1.0) val = 1.0;
    
    return val;
}

float randn(float mu, float sigma)
{
    float uni_noise = (float) rand()/RAND_MAX;
	float normal_noise = mu - (sigma*sqrt(3.0)/3.14159)*(logf(1.0-uni_noise) - logf(uni_noise));
	return normal_noise;
}
