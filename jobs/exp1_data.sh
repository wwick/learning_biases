for SIZE in 6 7 8
do
  for REWARD in 1 2 3 4 5 6
  do
    for EPISODE in 5 6 7 8
    do
      for HORIZON in 1 2 3
      do
        FILE="/scratch1/fs1/chien-ju.ho/RIS/518/scripts/${SIZE}_${REWARD}_${EPISODE}_${HORIZON}.npz"        
        if [ -f "$FILE" ]; then
            echo "$FILE exists."
        else 
            echo "$FILE does not exist."
            bsub -n 8 \
            -q general \
            -m general \
            -G compute-chien-ju.ho \
            -J ${SIZE} \
            -M 64GB \
            -N \
            -u saumik@wustl.edu \
            -o /home/n.saumik/518/learning_biases/jobs/exp1_data.%J \
            -R 'rusage[mem=64GB] span[hosts=1]' \
            -g /saumik/limit100 \
            -a "docker(saumikn/chesstrainer)" \
            "cd ~/518/learning_biases && /opt/conda/bin/python cse518_exp1.py" ${SIZE} ${REWARD} ${EPISODE} ${HORIZON}
        fi
      done
    done
  done
done