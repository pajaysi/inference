mlcr run-mlperf,inference,_find-performance,_full,_r5.1-dev \
   --model=dlrm-v2-99 \
   --implementation=reference \
   --framework=pytorch \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=cpu  \
   --docker --quiet \
   --test_query_count=10 --rerun