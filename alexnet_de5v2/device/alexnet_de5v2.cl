#include "../host/inc/alexnet.h"

__kernel 
__attribute((reqd_work_group_size(CONV_BLOCK_SIZE,CONV_BLOCK_SIZE,1)))
__attribute((num_simd_work_items(CONV_SIMD_ITEMS)))
void conv(__global short *restrict in_data,
		__global char *restrict conv_wt,
		__global char *restrict conv_bias,
		__global short *restrict out_data,
		unsigned wt_offset,		// group_no*number of input features per group*N_elem
		unsigned bias_offset,	// group_no*number of output featurs per group.
		unsigned in_offset,		// group_no*N_elem*padded(N_Fout_dim*N_Fout_dim)
		unsigned out_offset,	// group_no*number of output features per group*padded(N_Fout_dim*N_Fout_dim)
		unsigned N_elem,
		unsigned K_conv,
		unsigned S_conv,
		unsigned P_conv,
		unsigned N_Fin,			// Number of input features per group. Eg: 3 input channels for layer1, 48 input channels for layer 2
		unsigned N_Fin_dim,		// Number of pixels in input feature in (x/y direction). Eg. 227 pixels
		unsigned N_Fin_sq_pad,	// Special case when input is also a padded feature coming from another convolution block: padded(N_Fin_dim^2)
		unsigned N_Fout_dim		// Number of pixels in output feature in (x/y direction). Eg. 55 pixels
		)
{
	__local char weight[CONV_BLOCK_SIZE][CONV_BLOCK_SIZE];
    __local short input[CONV_BLOCK_SIZE][CONV_BLOCK_SIZE];
	
    // Block index
    unsigned block_y = get_group_id(1);

    // Local ID index (offset within a block)
    unsigned local_x = get_local_id(0);
    unsigned local_y = get_local_id(1);

	ushort global_y = get_global_id(1);
	ushort global_x = get_global_id(0);

	ushort K_conv_sq = K_conv*K_conv;

	// Compute loop bounds
    unsigned a_start = wt_offset + N_elem * CONV_BLOCK_SIZE * block_y;
    unsigned a_end   = a_start + N_elem - 1;

    int conv_out = (conv_bias[bias_offset+global_y])<<QN;
    
    for (int a = a_start, b = 0; a <= a_end; a += CONV_BLOCK_SIZE, b += CONV_BLOCK_SIZE)
    {
        weight[local_y][local_x] = conv_wt[a + N_elem * local_y + local_x];

   		ushort gx = b+local_y;
		ushort gy = global_x;
		ushort ch_no = gx/K_conv_sq;
		ushort i_k = gx-(ch_no*K_conv_sq);
		ushort k_row_no = i_k/K_conv;
		short k_row = k_row_no - P_conv;	//Padding
		short k_col = i_k - (k_row_no*K_conv) - P_conv;
		ushort out_feat_row = gy/N_Fout_dim;
		short row = (out_feat_row)*S_conv + k_row;
		short col = (gy-(out_feat_row*N_Fout_dim))*S_conv + k_col;
		unsigned location = in_offset + ch_no*N_Fin_sq_pad + row*N_Fin_dim + col;
		
		short data;
		if(gx > N_Fin*K_conv_sq || gy > N_Fout_dim*N_Fout_dim || 
			row<0 || col<0 || row >= N_Fin_dim || col >= N_Fin_dim)	// padding 
			data=0;
		else
			data = in_data[location];

		input[local_x][local_y] = data;

		// Wait for the entire block to be loaded.
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (ushort k = 0; k < CONV_BLOCK_SIZE; ++k)
        {
			short product = (weight[local_y][k] * input[local_x][k]);
            conv_out += (product>>QN);
        }

        // Wait for the block to be fully consumed before loading the next block.
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    out_data[out_offset + get_global_id(1) * get_global_size(0) + get_global_id(0)] = (conv_out>0)?conv_out>>(7-QN):0;
}

float get_norm(float);

__kernel 
__attribute((reqd_work_group_size(1,1,1)))
void norm_cross(__global short *restrict in_data, 
                        __global short *restrict out_data,
						unsigned N_in_sq,
						unsigned N_in_sq_pad,
						unsigned N_Fin
						)
{
	__local float sum_sq[55*55];
	for(unsigned i=0;i<N_in_sq;i++)
	{
		short temp1 = in_data[i];
		short temp2 = in_data[N_in_sq_pad+i];
		sum_sq[i] = ((temp1*temp1) + (temp2*temp2))>>(2*QN);
	}

	for(int i=0;i<N_Fin;i++)
	{
		for(unsigned j=0;j<N_in_sq;j+=NORM_BLOCK_SIZE)
		{
			#pragma unroll
			for(unsigned k=0;k<NORM_BLOCK_SIZE;k++)
			{
				short head = ((i+2) < N_Fin)?in_data[(i+2)*N_in_sq_pad+j+k]:0;		
				short tail = ((i-2) >= 0)?in_data[(i-2)*N_in_sq_pad+j+k]:0;
				if(j+k<N_in_sq)
				{
					sum_sq[j+k] += ((head*head)>>(2*QN));								// Add at head
					//out_data[i*N_in_sq+j+k]=in_data[i*N_in_sq_pad+j+k]/powr((1+alpha/LRN*sum_sq[j+k]),beta);
					out_data[i*N_in_sq+j+k]=in_data[i*N_in_sq_pad+j+k]*get_norm(alpha/LRN*sum_sq[j+k]);
					sum_sq[j+k] -= ((tail*tail)>>(2*QN));								// Subtract at tail
				}
			}
		}
	}
}

float get_norm(float a)
{
	if(a>=0.000000000000000000e+00 && a<2.770000000000000240e-01)
		return(1.000000000000000000e+00+(-0.604882853622783)*(a-0.000000000000000000e+00));
	else if(a>=2.770000000000000240e-01 && a<6.280000000000000027e-01)
		return(8.324474495464890822e-01+(-0.394893627647336)*(a-2.770000000000000240e-01));
	else if(a>=6.280000000000000027e-01 && a<1.074999999999999956e+00)
		return(6.938397862422742701e-01+(-0.2582303025983)*(a-6.280000000000000027e-01));
	else if(a>=1.074999999999999956e+00 && a<1.645000000000000018e+00)
		return(5.784108409808340623e-01+(-0.168881450436529)*(a-1.074999999999999956e+00));
	else if(a>=1.645000000000000018e+00 && a<2.371999999999999886e+00)
		return(4.821484142320127120e-01+(-0.110425834118896)*(a-1.645000000000000018e+00));
	else if(a>=2.371999999999999886e+00 && a<3.298999999999999932e+00)
		return(4.018688328275751287e-01+(-0.0721932314513022)*(a-2.371999999999999886e+00));
	else if(a>=3.298999999999999932e+00 && a<4.482000000000000206e+00)
		return(3.349457072722179518e-01+(-0.0471873554412027)*(a-3.298999999999999932e+00));
	else if(a>=4.482000000000000206e+00 && a<5.988999999999999879e+00)
		return(2.791230657852751817e-01+(-0.0308432289704369)*(a-4.482000000000000206e+00));
	else if(a>=5.988999999999999879e+00 && a<7.910000000000000142e+00)
		return(2.326423197268267240e-01+(-0.0201645538367102)*(a-5.988999999999999879e+00));
	else if(a>=7.910000000000000142e+00 && a<1.035999999999999943e+01)
		return(1.939062118065064122e-01+(-0.0131824278188802)*(a-7.910000000000000142e+00));
	else if(a>=1.035999999999999943e+01 && a<1.348300000000000054e+01)
		return(1.616092636502499402e-01+(-0.0086176203902491)*(a-1.035999999999999943e+01));
	else if(a>=1.348300000000000054e+01 && a<1.746499999999999986e+01)
		return(1.346964351715019825e-01+(-0.00563362140353829)*(a-1.348300000000000054e+01));
	else if(a>=1.746499999999999986e+01 && a<2.254200000000000159e+01)
		return(1.122633547426125228e-01+(-0.00368278111346424)*(a-1.746499999999999986e+01));
	else if(a>=2.254200000000000159e+01 && a<2.901399999999999935e+01)
		return(9.356587502955454605e-02+(-0.00240753872474281)*(a-2.254200000000000159e+01));
	else if(a>=2.901399999999999935e+01 && a<3.726700000000000301e+01)
		return(7.798428440301910514e-02+(-0.00157385484746763)*(a-2.901399999999999935e+01));
	else if(a>=3.726700000000000301e+01 && a<4.778800000000000381e+01)
		return(6.499526034686874121e-02+(-0.00102884342566653)*(a-3.726700000000000301e+01));
	else if(a>=4.778800000000000381e+01 && a<6.120199999999999818e+01)
		return(5.417079866543121625e-02+(-0.000672576705561931)*(a-4.778800000000000381e+01));
	else if(a>=6.120199999999999818e+01 && a<7.830299999999999727e+01)
		return(4.514885473702347218e-02+(-0.000439680645657729)*(a-6.120199999999999818e+01));
	else// if(a>=7.830299999999999727e+01 && a<9.999899999999999523e+01)
		return(3.762987601563065609e-02+(-0.000287701915626835)*(a-7.830299999999999727e+01));
}

__kernel 
__attribute((num_simd_work_items(POOL_SIMD_ITEMS)))
void max_pool(__global short *restrict in_data, 
                       __global short *restrict out_data,
					   unsigned N_in,
					   unsigned N_in_sq_pad,	// special case: when input is coming padded from convolution layer.
					   unsigned N_out)
{
    short index = get_global_id(0);
		
	for(int row=0;row<N_out;row++)
	{
		for(int col=0;col<N_out;col++)
		{
			short pool_out = -1000;
			#pragma unroll
			for(short j=0;j<K_pool*K_pool;j++)
			{
				uchar k_row = j/K_pool;
				uchar k_col = j-(k_row*K_pool);
				short new_data = in_data[index*N_in_sq_pad+(row*S_pool+k_row)*N_in+(col*S_pool+k_col)];
				if(pool_out<new_data)	// max_pool operation
					pool_out = new_data;
			}
			out_data[index*N_out*N_out+row*N_out+col]=pool_out;
		}
	}
}

__kernel 
__attribute((reqd_work_group_size(1,1,1)))
void innerproduct(__global short *restrict in_data, 
                        __global char *restrict ip_wt, 
						__global char *restrict ip_bias,
                        __global short *restrict out_data,
						unsigned N_in,
						unsigned N_out,
						char resolution,	// resolution: if 0 >>(7-QN) and if 1  >>(7-QN-QN_IP)
						char relu)			// bool not supported as kernel argument
{
	for(unsigned i=0;i<N_out;i++)
	{
		int ip_out = (ip_bias[i])<<QN;
		for(unsigned j=0;j<N_in;j+=IP_BLOCK_SIZE)
		{
			#pragma unroll 		
			for(ushort k=0;k<IP_BLOCK_SIZE;k++)
			{
				short product = (in_data[j+k]*ip_wt[i*N_in+j+k]);
				ip_out += (product>>QN);
			}
		}
		short ip_out_temp = (resolution==1)?ip_out>>(7-QN-QN_IP):ip_out>>(7-QN);
		out_data[i]=(relu==1 && ip_out<0)?0:ip_out_temp;
	}
}

__kernel 
__attribute((reqd_work_group_size(1,1,1)))
void predict_label(__global short *restrict in_data,
				   __global short *restrict out_data,
				   unsigned N_in,	// input dimensions i.e. length(fc8)
				   unsigned N_out)	// output top-k labels, k=N_out
{
	short max1_val=32767;
	
	for(uchar i=0;i<N_out;i++)
	{
		ushort max2_ind=0;
		short max2_val=in_data[0];
		for(ushort j=1;j<N_in;j++)
		{
			short new_data = in_data[j];
			if(new_data>max2_val && new_data<=max1_val)
			{	
				// if the values are equal, check if the index is already an output predicted label.
				bool present=0;
				for(uchar k=0;k<i;k++)
				{
					if(j==out_data[k])	
						present=1;
				}
				if(present==0)
				{
					max2_ind=j;
					max2_val=new_data;
				}
			}
		}
		out_data[i]=max2_ind;
		max1_val=max2_val;
	}
}
