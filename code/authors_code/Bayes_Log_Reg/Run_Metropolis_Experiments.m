for Dataset =  {'Australian' 'German' 'Heart' 'Pima' 'Ripley'}
    disp(['Running with ' Dataset{1} ' dataset'])
    cd('MCMC')
    BLR_metropolis(Dataset{1})
end