use crate::cuda::culib::{
    _halo2_multiopen_poly_calculation, _halo2_multiopen_poly_calculation_workspace_size,
    _halo2_multiopen_poly_max_len,
};
use crate::cuda::utils::{
    ensure_current_device_matches_ctx, query_device_free_bytes_for_chunking, FFITraitObject,
    HALO2_GPU_CTX,
};
use crate::cuda::HaloGpuError;
use ff::Field;
use openvm_cuda_common::d_buffer::DeviceBuffer;

// batch poly calculation for multiopen
pub fn batch_multiopen_poly_calculation_gpu<F: Field>(
    poly_in_many_ori: Vec<FFITraitObject>,
    poly_acc: &mut [F],
    poly_offset: usize,
    poly_length: usize,
    challenge_point: Vec<F>,
    evaluate_point: Vec<F>,
    evalaute_result: &mut [F],
) -> Result<(), HaloGpuError> {
    crate::perf_section!("basic_multiopen_poly_calc");
    ensure_current_device_matches_ctx()?;
    let batch_size = poly_in_many_ori.len();
    let poly_acc_obj = FFITraitObject::from_ref(&poly_acc[0]);
    let challenge_point_obj = FFITraitObject::from_ref(&challenge_point[0]);
    let evaluate_point_obj = FFITraitObject::from_ref(&evaluate_point[0]);
    let evaluate_result_obj = FFITraitObject::from_ref(&evalaute_result[0]);
    let scratch_bytes = unsafe {
        _halo2_multiopen_poly_calculation_workspace_size(poly_length as u64, batch_size as u64)
    } as usize;
    let scratch = DeviceBuffer::<u8>::with_capacity_on(scratch_bytes, &HALO2_GPU_CTX);
    let status = unsafe {
        _halo2_multiopen_poly_calculation(
            poly_in_many_ori.as_ptr(),
            &poly_acc_obj,
            poly_offset,
            poly_length,
            batch_size,
            &challenge_point_obj,
            &evaluate_point_obj,
            &evaluate_result_obj,
            scratch.as_mut_raw_ptr(),
            scratch_bytes as u64,
            HALO2_GPU_CTX.stream.as_raw(),
        )
    };

    if status.code != 0 {
        return Err(status.into());
    }
    Ok(())
}

// multiopen_poly_calculation: memory-aware + basic impl
pub fn multiopen_poly_calculation_gpu<F: Field>(
    poly_in_many_ori: Vec<FFITraitObject>,
    challenge_point: Vec<F>,
    poly_acc: &mut [F],
    evaluate_point: Vec<F>,
    evalaute_result: &mut [F],
    poly_length: usize,
) -> Result<(), HaloGpuError> {
    crate::perf_section!("multiopen_poly_calc");
    let batch_size = poly_in_many_ori.len();
    let max_length = unsafe {
        _halo2_multiopen_poly_max_len(
            poly_length,
            batch_size,
            query_device_free_bytes_for_chunking() as u64,
        )
    };
    if poly_length <= max_length {
        batch_multiopen_poly_calculation_gpu(
            poly_in_many_ori,
            poly_acc,
            0,
            poly_length,
            challenge_point,
            evaluate_point,
            evalaute_result,
        )
    } else {
        // max_length < poly_length
        // split ploy data into into chunks ( batch_size remains unchanged )
        let num_chunks = ((poly_length as f64) / (max_length as f64)).ceil() as usize;
        let chunk_size = max_length;
        log::debug!("poly_length: {} > max_length: {}", poly_length, max_length);
        log::debug!("num_chunks: {}, chunk_size: {}", num_chunks, chunk_size);
        let mut multi_eval_result: Vec<Vec<F>> = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            log::debug!("chunk_idx: {}", chunk_idx);
            let mut temp_result: Vec<F> = vec![F::ZERO; batch_size];
            let _offset = chunk_idx * chunk_size;
            let _lenght =
                if chunk_idx == num_chunks - 1 { poly_length - _offset } else { chunk_size };
            batch_multiopen_poly_calculation_gpu(
                poly_in_many_ori.clone(),
                poly_acc,
                _offset,
                _lenght,
                challenge_point.clone(),
                evaluate_point.clone(),
                &mut temp_result,
            )?;
            temp_result.iter_mut().enumerate().for_each(|(i, result)| {
                *result = (*result) * evaluate_point[i].pow_vartime([_offset as u64, 0, 0, 0])
            });
            multi_eval_result.push(temp_result);
        }
        evalaute_result.iter_mut().enumerate().for_each(|(i, result)| {
            *result = multi_eval_result.iter().fold(F::ZERO, |acc, res| acc + res[i]);
        });
        Ok(())
    }
}
