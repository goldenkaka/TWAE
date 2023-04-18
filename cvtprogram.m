function x = cvtprogram(dim_num,n)


  

  

      r = [];
%[ r, seed, it_num, it_diff, energy ] = cvt ( dim_num, n, batch, init, ...
%      sample, sample_num, it_max, it_fixed, seed, r );
   
    
    [ r, seed, it_num, it_diff, energy ] = cvt ( dim_num, n, 1000, 3, ...
      3, 10000, 50, 25, 123456789, r );
  x = {r,energy};
%
%  Write the data to a file.
%   
    save('cvt.mat','r')
    



  
end
 


function c = ch_cap ( c )

  if ( 'a' <= c && c <= 'z' )
    c = c + 'A' - 'a';
  end

  return
end
function [ r, seed, it_num, it_diff, energy ] = cvt ( dim_num, n, batch, init, ...
  sample, sample_num, it_max, it_fixed, seed, r )


  DEBUG = 1;

  if ( batch < 1 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'CVT - Fatal error!\n' );
    fprintf ( 1, '  The input value BATCH < 1.\n' );
    error ( 'CVT - Fatal error!' );
  end

  if ( seed <= 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'CVT - Fatal error!\n' );
    fprintf ( 1, '  The input value SEED <= 0.\n' );
    error ( 'CVT - Fatal error!' );
  end
  
  if ( DEBUG )
    fprintf ( 1, '\n' );
    fprintf ( 1, '  Step    SEED          L2-Change     Energy\n' );
    fprintf ( 1, '\n' );
  end
  
  it_num = 0;
  it_diff = 0.0;
  energy = 0.0;
  seed_init = seed;
%
%  Initialize the data unless the user has already done that.
%
  if ( init ~= 5 )

    initialize = 1;

    [ r, seed ] = cvt_sample ( dim_num, n, n, init, initialize, seed );

  end
  
  if ( DEBUG )
    fprintf ( 1, '  %4d  %12d\n', ...
      it_num, seed_init );
  end

%
%  If the initialization and sampling steps use the same random number
%  scheme, then the sampling scheme does not have to be initialized.
%
  if ( init == sample )
    initialize = 0;
  else
    initialize = 1;
  end
  
  while ( it_num < it_max )
%
%  If it's time to update the seed, save its current value
%  as the starting value for all iterations in this cycle.
%  If it's not time to update the seed, restore it to its initial
%  value for this cycle.
%
    if ( mod ( it_num, it_fixed ) == 0 )
      seed_base = seed;
    else
      seed = seed_base;
    end

    it_num = it_num + 1;

    seed_init = seed;

    [ r, seed, it_diff, energy ] = cvt_iterate ( dim_num, n, batch, sample, ...
      initialize, sample_num, seed, r );

    initialize = 0;

    if ( DEBUG )
      fprintf ( 1, '  %4d  %12d  %14e  %14e\n', ...
        it_num, seed_init, it_diff, energy );
    end

  end 

  return
end
function [ r, seed, it_diff, energy ] = cvt_iterate ( dim_num, n, batch, ...
  sample, initialize, sample_num, seed, r )


  energy = 0.0;
  r2(1:dim_num,1:n) = r(1:dim_num,1:n);
  count(1:n) = 1;
%
%  Generate the sampling points S in batches.
%
  have = 0;

  while ( have < sample_num )

    get = min ( sample_num - have, batch );

    [ s, seed ] = cvt_sample ( dim_num, sample_num, get, sample, initialize, ...
      seed );
  
    initialize = 0;
    have = have + get;
%
%  Find the index N of the nearest cell generator to each sample point S.
%
    nearest = find_closest ( dim_num, n, get, s, r );
%
%  Add S to the centroid associated with generator N.
%
    for j = 1 : get
      r2(1:dim_num,nearest(j)) = r2(1:dim_num,nearest(j)) + s(1:dim_num,j);
      energy = energy + sum ( ( r(1:dim_num,nearest(j)) - s(1:dim_num,j) ).^2 );
      count(nearest(j)) = count(nearest(j)) + 1;
    end

  end
%
%  Estimate the centroids.
%
  for j = 1 : n
    r2(1:dim_num,j) = r2(1:dim_num,j) / count(j);
  end
%
%  Determine the sum of the distances between generators and centroids.
%
  it_diff = 0.0;
  for j = 1 : n
    it_diff = it_diff + sqrt ( sum ( ( r2(1:dim_num,j) - r(1:dim_num,j) ).^2 ) );
  end
%
%  Replace the generators by the centroids.
%
  r(1:dim_num,1:n) = r2(1:dim_num,1:n);
%
%  Normalize the discrete energy estimate.
%
  energy = energy / sample_num;

  return
end
function [ r, seed ] = cvt_sample ( dim_num, n, n_now, sample, initialize, seed )


  if ( n_now < 1 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'CVT_SAMPLE - Fatal error!\n' );
    fprintf ( 1, '  N_NOW < 1.\n' );
    error ( 'CVT_SAMPLE - Fatal error!' );
  end

  if ( sample == -1 )

    if ( initialize )
      random_initialize ( seed );
    end

    r = rand ( dim_num, n_now );

    seed = seed + n_now * dim_num;

  elseif ( sample == 0 )

    [ r, seed ] = r8mat_uniform_01 ( dim_num, n_now, seed );

  elseif ( sample == 1 )

    halton_step = seed;
    halton_seed(1:dim_num) = 0;
    halton_leap(1:dim_num) = 1;

    for i = 1 : dim_num
      halton_base(i) = prime ( i );
    end

    r(1:dim_num,1:n_now) = i4_to_halton_sequence ( dim_num, n_now, ...
      halton_step, halton_seed, halton_leap, halton_base );

    seed = seed + n_now;

  elseif ( sample == 2 )

    exponent = 1.0 / dim_num;
    ngrid = floor ( n^exponent );
    rank_max = ngrid^dim_num;

    if ( rank_max < n )
      ngrid = ngrid + 1;
      rank_max = ngrid^dim_num;
    end

    if ( initialize )
      rank = -1;
      tuple(1:dim_num) = tuple_next_fast ( ngrid, dim_num, rank );
    end

    rank = mod ( seed, rank_max );

    for j = 1 : n_now
      tuple(1:dim_num) = tuple_next_fast ( ngrid, dim_num, rank );
      rank = rank + 1;
      rank = mod ( rank, rank_max );
      r(1:dim_num,j) = ( 2 * tuple(1:dim_num)' - 1 ) / ( 2 * ngrid );
    end

    seed = seed + n_now;

  elseif ( sample == 3 )

    [ r, seed ] = user1 ( dim_num, n_now, seed );
    
  elseif ( sample == 4 )

    [ r, seed ] = user2 ( dim_num, n_now, seed );

  else

    fprintf ( 1, '\n' );
    fprintf ( 1, 'CVT_SAMPLE - Fatal error!\n' );
    fprintf ( 1, '  The value of SAMPLE = %d is illegal.\n', sample );
    error ( 'CVT_SAMPLE - Fatal error!' );

  end

  return
end
function [ r, seed ] = user1 ( dim_num, n_now, seed )
  x = randn(dim_num, n_now);
  norm = sqrt(sum(x.^2,1));
  radius = rand(1, n_now).^(1.0/dim_num);
  r = x./norm.*radius;
  %r = x./norm;
end
function [ r, seed ] = user2 ( dim_num, n_now, seed )
  a = ones(1,dim_num);
  p = length(a);
  r = gamrnd(repmat(a,n_now,1),1,[n_now,p]);
  r = r ./ repmat(sum(r,2),1,p);
  r = r.';
end
function table = data_read ( input_filename, m, n )


  string = ' ';

  for i = 0 : m
    string = strcat ( string, ' %f' );
  end

  input_unit = fopen ( input_filename );

  if ( input_unit < 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'DATA_READ - Error!\n' );
    fprintf ( 1, '  Could not open the input file.\n' );
    error ( 'DATA_READ - Error!' );
    return;
  end

  i = 0;

  while ( i < n )

    line = fgets ( input_unit );

    if ( line == -1 )
      fprintf ( 1, '\n' );
      fprintf ( 1, 'DATA_READ - Error!\n' );
      fprintf ( 1, '  End of input while reading data.\n' );
      error ( 'DATA_READ - Error!' );
    end

    if ( line(1) == '#' )

    elseif ( s_len_trim ( line ) == 0 )
      
    else

      [ x, count ] = sscanf ( line, string );

      if ( count == m )
        i = i + 1;
        table(1:m,i) = x(1:m);
      end

    end

  end

  fclose ( input_unit );

  return
end
function nearest = find_closest ( dn, gn, sn, s, g )

  ones_k = ones ( 1, gn );
  nearest = NaN ( 1, sn );

  for i = 1 : sn
    d1(1:dn,1:gn) = g(1:dn,1:gn) - s(1:dn,i) * ones_k;
    d2 = sum ( d1 .* d1, 1 );
    [ min_val, min_loc ] = min ( d2 );
    nearest(i) = min_loc;
  end

  return
end
function value = halham_dim_num_check ( dim_num )

  if ( dim_num < 1 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'HALHAM_DIM_NUM_SET - Fatal error!\n' );
    fprintf ( 1, '  Input value of DIM_NUM < 1!\n' );
    fprintf ( 1, '  DIM_NUM = %d\n', dim_num );
    value = 0;
  else
    value = 1;
  end

  return
end
function value = halham_leap_check ( dim_num, leap )

  if ( any ( leap(1:dim_num) < 1 ) )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'HALHAM_LEAP_CHECK - Fatal error!\n' );
    fprintf ( 1, '  At least one of the input leap entries is <= 1!\n' );
    value = 0;
  else
    value = 1;
  end

  return
end
function value = halham_n_check ( n )

  if ( n < 1 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'HALHAM_N_CHECK - Fatal error!\n' );
    fprintf ( 1, '  Input value of N < 1!\n' );
    fprintf ( 1, '  N = %d\n', n );
    value = 0;
  else
    value = 1;
  end

  return
end
function value = halham_seed_check ( dim_num, seed )

  if ( any ( seed(1:dim_num) < 0 ) )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'HALHAM_SEED_CHECK - Fatal error!\n' );
    fprintf ( 1, '  At least one of the input seeds is < 0!\n' );
    value = 0;
  else
    value = 1;
  end

  return
end
function value = halham_step_check ( step )

  if ( step < 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'HALHAM_STEP_CHECK - Fatal error!\n' );
    fprintf ( 1, '  Input value of STEP < 0!\n' );
    fprintf ( 1, '  STEP = %d\n', step );
    value = 0;
  else
    value = 1;
  end

  return
end
function value = halton_base_check ( dim_num, base )

  if ( any ( base(1:dim_num) <= 1 ) )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'HALTON_BASE_CHECK - Fatal error!\n' );
    fprintf ( 1, '  At least one of the input bases is <= 1!\n' );
    value = 0;
  else
    value = 1;
  end

  return
end
function r = i4_to_halton_sequence ( dim_num, n, step, seed, leap, base )

  dim_num = floor ( dim_num );
  n = floor ( n );
  step = floor ( step );
  seed(1:dim_num) = floor ( seed(1:dim_num) );
  leap(1:dim_num) = floor ( leap(1:dim_num) );
  base(1:dim_num) = floor ( base(1:dim_num) );
%
%  Check the input.
%
  if ( ~halham_dim_num_check ( dim_num ) )
    error ( 'I4_TO_HALTON_SEQUENCE - Fatal error!' );
  end

  if ( ~halham_n_check ( n ) )
    error ( 'I4_TO_HALTON_SEQUENCE - Fatal error!' );
  end

  if ( ~halham_step_check ( step ) )
    error ( 'I4_TO_HALTON_SEQUENCE - Fatal error!' );
  end

  if ( ~halham_seed_check ( dim_num, seed ) )
    error ( 'I4_TO_HALTON_SEQUENCE - Fatal error!' );
  end

  if ( ~halham_leap_check ( dim_num, leap ) )
    error ( 'I4_TO_HALTON_SEQUENCE - Fatal error!' );
  end

  if ( ~halton_base_check ( dim_num, base ) )
    error ( 'I4_TO_HALTON_SEQUENCE - Fatal error!' );
  end
%
%  Calculate the data.
%
  r(1:dim_num,1:n) = 0.0;
  
  for i = 1: dim_num

    seed2(1:n) = seed(i) + step * leap(i) : leap(i) : ...
                 seed(i) + ( step + n - 1 ) * leap(i);

    base_inv = 1.0 / base(i);
  
    while ( any ( seed2 ~= 0 ) )
      digit(1:n) = mod ( seed2(1:n), base(i) );
      r(i,1:n) = r(i,1:n) + digit(1:n) * base_inv;
      base_inv = base_inv / base(i);
      seed2(1:n) = floor ( seed2(1:n) / base(i) );
    end

  end

  return
end
function p = prime ( n )

  prime_max = 1600;

  prime_vector(1:1600) = [
        2,    3,    5,    7,   11,   13,   17,   19,   23,   29, ...
       31,   37,   41,   43,   47,   53,   59,   61,   67,   71, ...
       73,   79,   83,   89,   97,  101,  103,  107,  109,  113, ...
      127,  131,  137,  139,  149,  151,  157,  163,  167,  173, ...
      179,  181,  191,  193,  197,  199,  211,  223,  227,  229, ...
      233,  239,  241,  251,  257,  263,  269,  271,  277,  281, ...
      283,  293,  307,  311,  313,  317,  331,  337,  347,  349, ...
      353,  359,  367,  373,  379,  383,  389,  397,  401,  409, ...
      419,  421,  431,  433,  439,  443,  449,  457,  461,  463, ...
      467,  479,  487,  491,  499,  503,  509,  521,  523,  541, ...
      547,  557,  563,  569,  571,  577,  587,  593,  599,  601, ...
      607,  613,  617,  619,  631,  641,  643,  647,  653,  659, ...
      661,  673,  677,  683,  691,  701,  709,  719,  727,  733, ...
      739,  743,  751,  757,  761,  769,  773,  787,  797,  809, ...
      811,  821,  823,  827,  829,  839,  853,  857,  859,  863, ...
      877,  881,  883,  887,  907,  911,  919,  929,  937,  941, ...
      947,  953,  967,  971,  977,  983,  991,  997, 1009, 1013, ...
     1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, ...
     1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, ...
     1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, ...
     1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, ...
     1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, ...
     1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, ...
     1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, ...
     1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, ...
     1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, ...
     1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, ...
     1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, ...
     1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, ...
     1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, ...
     1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, ...
     2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, ...
     2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, ...
     2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, ...
     2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, ...
     2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, ...
     2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, ...
     2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, ...
     2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, ...
     2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, ...
     2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, ...
     2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, ...
     2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, ...
     3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, ...
     3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, ...
     3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, ...
     3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, ...
     3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, ...
     3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, ...
     3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, ...
     3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, ...
     3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, ...
     3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, ...
     3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, ...
     3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, ...
     4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, ...
     4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, ...
     4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, ...
     4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, ...
     4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, ...
     4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, ...
     4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, ...
     4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, ...
     4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, ...
     4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, ...
     4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, ...
     4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, ...
     5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, ...
     5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, ...
     5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, ...
     5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, ...
     5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, ...
     5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, ...
     5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, ...
     5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, ...
     5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, ...
     5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, ...
     5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, ...
     5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, ...
     6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, ...
     6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, ...
     6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, ...
     6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, ...
     6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, ...
     6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, ...
     6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, ...
     6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, ...
     6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, ...
     6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, ...
     6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, ...
     7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, ...
     7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, ...
     7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, ...
     7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, ...
     7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, ...
     7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, ...
     7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, ...
     7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, ...
     7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, ...
     7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, ...
     7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, ...
     8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, ...
     8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, ...
     8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, ...
     8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, ...
     8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, ...
     8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, ...
     8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, ...
     8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, ...
     8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, ...
     8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, ...
     8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, ...
     9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, ...
     9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, ...
     9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, ...
     9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, ...
     9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, ...
     9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, ...
     9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, ...
     9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, ...
     9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, ...
     9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, ...
     9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973,10007, ...
    10009,10037,10039,10061,10067,10069,10079,10091,10093,10099, ...
    10103,10111,10133,10139,10141,10151,10159,10163,10169,10177, ...
    10181,10193,10211,10223,10243,10247,10253,10259,10267,10271, ...
    10273,10289,10301,10303,10313,10321,10331,10333,10337,10343, ...
    10357,10369,10391,10399,10427,10429,10433,10453,10457,10459, ...
    10463,10477,10487,10499,10501,10513,10529,10531,10559,10567, ...
    10589,10597,10601,10607,10613,10627,10631,10639,10651,10657, ...
    10663,10667,10687,10691,10709,10711,10723,10729,10733,10739, ...
    10753,10771,10781,10789,10799,10831,10837,10847,10853,10859, ...
    10861,10867,10883,10889,10891,10903,10909,10937,10939,10949, ...
    10957,10973,10979,10987,10993,11003,11027,11047,11057,11059, ...
    11069,11071,11083,11087,11093,11113,11117,11119,11131,11149, ...
    11159,11161,11171,11173,11177,11197,11213,11239,11243,11251, ...
    11257,11261,11273,11279,11287,11299,11311,11317,11321,11329, ...
    11351,11353,11369,11383,11393,11399,11411,11423,11437,11443, ...
    11447,11467,11471,11483,11489,11491,11497,11503,11519,11527, ...
    11549,11551,11579,11587,11593,11597,11617,11621,11633,11657, ...
    11677,11681,11689,11699,11701,11717,11719,11731,11743,11777, ...
    11779,11783,11789,11801,11807,11813,11821,11827,11831,11833, ...
    11839,11863,11867,11887,11897,11903,11909,11923,11927,11933, ...
    11939,11941,11953,11959,11969,11971,11981,11987,12007,12011, ...
    12037,12041,12043,12049,12071,12073,12097,12101,12107,12109, ...
    12113,12119,12143,12149,12157,12161,12163,12197,12203,12211, ...
    12227,12239,12241,12251,12253,12263,12269,12277,12281,12289, ...
    12301,12323,12329,12343,12347,12373,12377,12379,12391,12401, ...
    12409,12413,12421,12433,12437,12451,12457,12473,12479,12487, ...
    12491,12497,12503,12511,12517,12527,12539,12541,12547,12553, ...
    12569,12577,12583,12589,12601,12611,12613,12619,12637,12641, ...
    12647,12653,12659,12671,12689,12697,12703,12713,12721,12739, ...
    12743,12757,12763,12781,12791,12799,12809,12821,12823,12829, ...
    12841,12853,12889,12893,12899,12907,12911,12917,12919,12923, ...
    12941,12953,12959,12967,12973,12979,12983,13001,13003,13007, ...
    13009,13033,13037,13043,13049,13063,13093,13099,13103,13109, ...
    13121,13127,13147,13151,13159,13163,13171,13177,13183,13187, ...
    13217,13219,13229,13241,13249,13259,13267,13291,13297,13309, ...
    13313,13327,13331,13337,13339,13367,13381,13397,13399,13411, ...
    13417,13421,13441,13451,13457,13463,13469,13477,13487,13499 ];

  if ( n == -1 )
    p = prime_max;
  elseif ( n == 0 )
    p = 1;
  elseif ( n <= prime_max )
    p = prime_vector(n);
  else
    p = -1;
  end

  return
end
function [ r, seed ] = r8mat_uniform_01 ( m, n, seed )

  r = zeros ( m, n );

  i4_huge = 2147483647;

  if ( seed == 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'R8MAT_UNIFORM_01 - Fatal error!\n' );
    fprintf ( 1, '  Input SEED = 0!\n' );
    error ( 'R8MAT_UNIFORM_01 - Fatal error!' );
  end

  for j = 1 : n
    for i = 1 : m

      seed = floor ( seed );

      seed = mod ( seed, i4_huge );

      if ( seed < 0 ) 
        seed = seed + i4_huge;
      end 

      k = floor ( seed / 127773 );

      seed = 16807 * ( seed - k * 127773 ) - k * 2836;

      if ( seed < 0 )
        seed = seed + i4_huge;
      end

      r(i,j) = seed * 4.656612875E-10;

    end
  end

  return
end
function r8mat_write ( output_filename, m, n, table )


%
%  Open the file.
%
  output_unit = fopen ( output_filename, 'wt' );

  if ( output_unit < 0 ) 
    fprintf ( 1, '\n' );
    fprintf ( 1, 'R8MAT_WRITE - Error!\n' );
    fprintf ( 1, '  Could not open the output file.\n' );
    error ( 'R8MAT_WRITE - Error!' );
  end
%
%  Write the data.
%
%  For smaller data files, and less precision, try:
%
%     fprintf ( output_unit, '  %14.6e', table(i,j) );
%
  for j = 1 : n
    for i = 1 : m
      fprintf ( output_unit, '  %24.16e', table(i,j) );
    end
    fprintf ( output_unit, '\n' );
  end
%
%  Close the file.
%
  fclose ( output_unit );

  return
end
function seed = random_initialize ( seed )

  debug = 0;

  if ( seed ~= 0 )

    if ( debug )
      fprintf ( 1, '\n' );
      fprintf ( 1, 'RANDOM_INITIALIZE:\n' );
      fprintf ( 1, '  Initialize RANDOM_NUMBER, user SEED = %d\n', seed );
    end

  else

    seed = get_seed ( seed );

    if ( debug )
      fprintf ( 1, '\n' );
      fprintf ( 1, 'RANDOM_INITIALIZE:\n' );
      fprintf ( 1, '  Initialize RANDOM_NUMBER, arbitrary SEED = %d\n', seed );
    end

  end

  rand ( 'state', seed );

  return
end
function value = s_eqi ( s1, s2 )


  FALSE = 0;
  TRUE = 1;

  len1 = length ( s1 );
  len2 = length ( s2 );
  lenc = min ( len1, len2 );

  for i = 1 : lenc

    c1 = ch_cap ( s1(i) );
    c2 = ch_cap ( s2(i) );

    if ( c1 ~= c2 )
      value = FALSE;
      return
    end

  end

  for i = lenc + 1 : len1
    if ( s1(i) ~= ' ' )
      value = FALSE;
      return
    end
  end

  for i = lenc + 1 : len2
    if ( s2(i) ~= ' ' )
      value = FALSE;
      return
    end
  end

  value = TRUE;

  return
end
function len = s_len_trim ( s )


  len = length ( s );

  while ( 0 < len )
    if ( s(len) ~= ' ' )
      return
    end
    len = len - 1;
  end

  return
end
function timestamp ( )


  t = now;
  c = datevec ( t );
  s = datestr ( c, 0 );
  fprintf ( 1, '%s\n', s );

  return
end
function x = tuple_next_fast ( m, n, rank )

  global tuple_next_fast_BASE

  if ( rank < 0 )

    tuple_next_fast_BASE(n) = 1;
    for i = n-1 : -1 : 1
      tuple_next_fast_BASE(i) = tuple_next_fast_BASE(i+1) * m;
    end

    x(1:n) = -1;

  else

    x(1:n) = mod ( floor ( rank ./ tuple_next_fast_BASE(1:n) ), m ) + 1;

  end

  return
end