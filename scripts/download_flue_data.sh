#!/bin/bash

TASKS=("CLS-Books" "CLS-DVD" "CLS-Music" "PAWSX" "XNLI" "Parsing-Dep" "Parsing-Const" "WSD-Verb" "WSD-Nouns")

while getopts "d:t:" OPTION; do
  case $OPTION in
  d)
    DATADIR=$OPTARG
    ;;
  t)
    TASKS=($OPTARG)
    ;;
  *)
    echo "Incorrect options provided"
    exit 1
    ;;
  esac
done

[ ! -d $DATADIR ] && echo "Output data directory $DATADIR does not exists." && exit 1;
git clone https://github.com/getalp/Flaubert.git $DATADIR/Flaubert &> /dev/null
chmod -R +x $DATADIR/Flaubert/flue
export PYTHONPATH=$PYTHONPATH:$DATADIR/Flaubert

for task_ in ${TASKS[*]}; do
  arrIN=(${task_//-/ })
  task=${arrIN[0]}
  subtask=${arrIN[1]}
  # echo "*** Downloading data from $task $subtask task ***"
  if [ $task = "WSD" ]; then
    if [ $subtask = "Verbs" ]; then
      ./spinner.sh "Downloading $task $subtask" wget http://www.llf.cnrs.fr/dataset/fse/FSE-1.1-10_12_19.tar.gz -O $DATADIR/FSE-1.1-10_12_19.tar.gz
      tar -xzf $DATADIR/FSE-1.1-10_12_19.tar.gz -C $DATADIR && rm $DATADIR/FSE-1.1-10_12_19.tar.gz &> /dev/null
      if [ ! -d $DATADIR/WSD ]; then
        mkdir $DATADIR/WSD
      fi
      mv $DATADIR/FSE-1.1-191210 $DATADIR/WSD/Verbs
      ./spinner.sh "Preprocessing $task $subtask" python $DATADIR/Flaubert/flue/wsd/verbs/prepare_data.py --data $DATADIR/WSD/Verbs --output $DATADIR/WSD/Verbs/
      rm $DATADIR/WSD/Verbs/FSE-*
      rm $DATADIR/WSD/Verbs/wiktionary*
    elif [ $subtask = "Nouns" ]; then
      if [ ! -d $DATADIR/WSD ]; then
        mkdir $DATADIR/WSD
      fi
      ./spinner.sh "Downloading $task $subtask" $DATADIR/Flaubert/flue/wsd/nouns/0.get_data.sh
      DISAMBIGUATE_GIT_URL=https://github.com/getalp/disambiguate.git
      UFSAC_GIT_URL=https://github.com/getalp/UFSAC.git
      DISAMBIGUATE_TARGET_DIRECTORY=$DATADIR/Flaubert/flue/wsd/nouns/disambiguate
      UFSAC_TARGET_DIRECTORY=$DATADIR/Flaubert/flue/wsd/nouns/disambiguate/UFSAC
      rm -rf $DISAMBIGUATE_TARGET_DIRECTORY
      rm -rf $UFSAC_TARGET_DIRECTORY
      git clone $DISAMBIGUATE_GIT_URL $DISAMBIGUATE_TARGET_DIRECTORY &> /dev/null
      git clone $UFSAC_GIT_URL $UFSAC_TARGET_DIRECTORY &> /dev/null
      $UFSAC_TARGET_DIRECTORY/java/install.sh &> /dev/null
      $DISAMBIGUATE_TARGET_DIRECTORY/java/compile.sh &> /dev/null
      ./spinner.sh "Preprocessing $task $subtask" $DATADIR/Flaubert/flue/wsd/nouns/2.prepare_data.sh
      mv $DATADIR/Flaubert/flue/wsd/nouns/prepared_data $DATADIR/WSD/Nouns
      mv $DATADIR/Flaubert/flue/wsd/nouns/corpus/*.xml $DATADIR/WSD/Nouns/
    fi
  elif [ $task = "Parsing" ]; then
    echo -e "\e[1m\e[31mWarning: Downloading Parsing task is not handled. Please visit https://dokufarm.phil.hhu.de/spmrl2014/ for manual downloads.\e[0m\e[0m"
  elif [ $task = "CLS" ]; then
    if [ ! -d $DATADIR/$task_ ] ; then
      ./spinner.sh "Downloading $task" $DATADIR/Flaubert/flue/get-data-${task,,}.sh $DATADIR/$task
      ./spinner.sh "Preprocessing $task" python $DATADIR/Flaubert/flue/extract_split_cls.py --indir $DATADIR/CLS/raw/cls-acl10-unprocessed --outdir $DATADIR/CLS/processed --do_lower False
      for sub_task in 'Books' 'DVD' 'Music'; do
        test -d $DATADIR/$task"-"$sub_task || mkdir $DATADIR/$task"-"$sub_task
        mv $DATADIR/$task/processed/${sub_task,,}/valid_0.tsv $DATADIR/$task"-"$sub_task/dev.tsv
        mv $DATADIR/$task/processed/${sub_task,,}/test_0.tsv $DATADIR/$task"-"$sub_task/test.tsv
        mv $DATADIR/$task/processed/${sub_task,,}/train_0.tsv $DATADIR/$task"-"$sub_task/train.tsv
      done
      rm -r $DATADIR/$task/
    fi
    # mv $DATADIR/$task/raw/cls-acl10-unprocessed/fr/* $DATADIR/$task/raw
    # rm -r $DATADIR/$task/raw/cls-acl10-unprocessed
    # rm $DATADIR/$task/raw/cls-acl10-unprocessed.tar.gz
  elif [ $task = "PAWSX" ]; then
    ./spinner.sh "Downloading $task $subtask" $DATADIR/Flaubert/flue/get-data-${task,,}.sh  $DATADIR/$task
    ./spinner.sh "Preprocessing $task $subtask" python $DATADIR/Flaubert/flue/extract_pawsx.py --indir $DATADIR/$task/raw/x-final --outdir $DATADIR/$task --do_lower False
    # rm $DATADIR/$task/raw/x-final.tar.gz
    # mv $DATADIR/$task/raw/x-final/fr/* $DATADIR/$task/raw
    # mv $DATADIR/$task/x-final/processed $DATADIR/$task/processed
    # rm -r  $DATADIR/$task/raw/x-final
    # rm -r  $DATADIR/$task/x-final
    # mv $DATADIR/$task/raw/dev_2k.tsv $DATADIR/$task/raw/dev.tsv
    # mv $DATADIR/$task/raw/test_2k.tsv $DATADIR/$task/raw/test.tsv
    # mv $DATADIR/$task/raw/translated_train.tsv $DATADIR/$task/raw/train.tsv
    mv $DATADIR/$task/valid_0.tsv $DATADIR/$task/dev.tsv
    mv $DATADIR/$task/test_0.tsv $DATADIR/$task/test.tsv
    mv $DATADIR/$task/train_0.tsv $DATADIR/$task/train.tsv
    rm -r  $DATADIR/$task/raw/
    # rm -r  $DATADIR/$task/processed/
  elif [ $task = "XNLI" ]; then
    ./spinner.sh "Downloading $task $subtask" $DATADIR/Flaubert/flue/get-data-${task,,}.sh  $DATADIR/$task
    ./spinner.sh "Preprocessing $task $subtask" python $DATADIR/Flaubert/flue/extract_xnli.py --indir $DATADIR/$task/processed/ --do_lower False
    mv $DATADIR/$task/processed/valid_0.xlm.tsv $DATADIR/$task/dev.tsv
    mv $DATADIR/$task/processed/test_0.xlm.tsv $DATADIR/$task/test.tsv
    mv $DATADIR/$task/processed/train_0.xlm.tsv $DATADIR/$task/train.tsv
    rm -r  $DATADIR/$task/raw/
    rm -r  $DATADIR/$task/processed/
  else
    ./spinner.sh "Downloading $task $subtask" $DATADIR/Flaubert/flue/get-data-${task,,}.sh  $DATADIR/$task
  fi
done

rm -rf $DATADIR/Flaubert

exit 0;
