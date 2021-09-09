#!/bin/bash
# Shows a spinner while another command is running. Randomly picks one of 12 spinner styles.
# @args command to run (with any parameters) while showing a spinner.
#       E.g. ‹spinner sleep 10›
# https://unix.stackexchange.com/questions/225179/display-spinner-while-waiting-for-some-process-to-finish/565551

blue=`tput setaf 4`
bold=`tput bold`
reset=`tput sgr0`

function shutdown() {
  tput cnorm # reset cursor
}
trap shutdown EXIT

function spinner() {
  # make sure we use non-unicode character type locale
  # (that way it works for any locale as long as the font supports the characters)
  local LC_CTYPE=C

  local pid=$1 # Process Id of the previous running command
  local desc="${@:2}"


  local spin='⣾⣽⣻⢿⡿⣟⣯⣷'
  local charwidth=3

  local i=0
  tput civis # cursor invisible
  while kill -0 $pid 2>/dev/null; do
    local i=$(((i + $charwidth) % ${#spin}))

    printf "\r%s%s%s %s" "${blue}${bold}" "${spin:$i:$charwidth}" "${reset}" "$desc"
    sleep .1
  done
  printf "\n"
  tput cnorm
  wait $pid # capture exit code
  return $?
}

("${@:2}" &> /dev/null) &
spinner $! $1