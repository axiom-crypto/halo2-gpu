uint32_t k = log_n;
uint32_t batch_size = 9; // max_threads = 1024
uint32_t combine_size_1 = 0;
uint32_t combine_size_2 = 0;

// devide log_n iterations to [batch_size | batch_size | ... | log_n % batch_size] several batches
// each GPU block calculate 2^(batch_size + combine_size_1) inputs
// if log_n % batch_size != 0, each GPU block in the last batch caculate 2^(log_n % batch_size + combine_size_2) inputs
// note:
// 0 < batch_size + combine_size_1 <= Field::NTT_BLOCK_K
// 0 < log_n % batch_size + combine_size_2 <= Field::NTT_BLOCK_K
// when Field::LIMBS, Field::WIDTH and GPU card change, need to reprofile the following configuration
{
    switch (k) {
    case 7:
        batch_size = 4;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 8:
        batch_size = 4;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 9:
        batch_size = 5;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 10:
        batch_size = 5;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 11:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 12:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 13:
        batch_size = 7;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 14:
        batch_size = 7;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 15:
        batch_size = 5;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 16:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 2;
        break;
    case 17:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 18:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 19:
        batch_size = 5;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 20:
        batch_size = 7;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 21:
        batch_size = 7;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 22:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 2;
        break;
    case 23:
        // Reprofiled for sm_120 (RTX 5090): batch_size 8 does the 23 levels in
        // 3 global passes vs 4, cutting ntt_dit ~16%. Larger widths regress on
        // occupancy here; confirm on other GPUs (see header note).
        batch_size = 8;
        combine_size_1 = 0;
        combine_size_2 = 1;
        break;
    case 24:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 25:
        batch_size = 5;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    case 26:
        batch_size = 7;
        combine_size_1 = 0;
        combine_size_2 = 2;
        break;
    case 27:
        batch_size = 6;
        combine_size_1 = 0;
        combine_size_2 = 3;
        break;
    case 28:
        batch_size = 7;
        combine_size_1 = 0;
        combine_size_2 = 0;
        break;
    default:
        break;
    }
}