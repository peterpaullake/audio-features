'''
  This is a custom version of absl.app that doesn't call
  sys.exit. We use this because DeepSpeech uses Abseil for
  handling command line arguments, but Abseil calls sys.exit,
  which doesn't work in Jupyter Notebook.
'''

import absl.app

def _run_main(main, argv):
  """Calls main, optionally with pdb or profiler."""
  if absl.app.FLAGS.run_with_pdb:
    absl.app.sys.exit(absl.app.pdb.runcall(main, argv))
  elif absl.app.FLAGS.run_with_profiling or absl.app.FLAGS.profile_file:
    # Avoid import overhead since most apps (including performance-sensitive
    # ones) won't be run with profiling.
    import atexit
    if absl.app.FLAGS.use_cprofile_for_profiling:
      import cProfile as profile
    else:
      import profile
    profiler = profile.Profile()
    if absl.app.FLAGS.profile_file:
      atexit.register(profiler.dump_stats, absl.app.FLAGS.profile_file)
    else:
      atexit.register(profiler.print_stats)
    retval = profiler.runcall(main, argv)
    # absl.app.sys.exit(retval)
  else:
    main(argv)
    # absl.app.sys.exit(main(argv))

def run(
    main,
    argv=None,
    flags_parser=absl.app.parse_flags_with_usage,
):
  """Begins executing the program.

  Args:
    main: The main function to execute. It takes an single argument "argv",
        which is a list of command line arguments with parsed flags removed.
        If it returns an integer, it is used as the process's exit code.
    argv: A non-empty list of the command line arguments including program name,
        sys.argv is used if None.
    flags_parser: Callable[[List[Text]], Any], the function used to parse flags.
        The return value of this function is passed to `main` untouched.
        It must guarantee FLAGS is parsed after this function is called.
  - Parses command line flags with the flag module.
  - If there are any errors, prints usage().
  - Calls main() with the remaining arguments.
  - If main() raises a UsageError, prints usage and the error message.
  """
  try:
    args = absl.app._run_init(
        sys.argv if argv is None else argv,
        flags_parser,
    )
    while absl.app._init_callbacks:
      callback = absl.app._init_callbacks.popleft()
      callback()
    try:
      # absl.app._run_main(main, args)
      _run_main(main, args)
    except absl.app.UsageError as error:
      absl.app.usage(shorthelp=True, detailed_error=error, exitcode=error.exitcode)
    except:
      exc = absl.app.sys.exc_info()[1]
      # Don't try to post-mortem debug successful SystemExits, since those
      # mean there wasn't actually an error. In particular, the test framework
      # raises SystemExit(False) even if all tests passed.
      if isinstance(exc, SystemExit) and not exc.code:
        raise

      # Check the tty so that we don't hang waiting for input in an
      # non-interactive scenario.
      if FLAGS.pdb_post_mortem and sys.stdout.isatty():
        absl.app.absl.app.traceback.print_exc()
        print()
        print(' *** Entering post-mortem debugging ***')
        print()
        absl.app.pdb.post_mortem()
      raise
  except Exception as e:
    absl.app._call_exception_handlers(e)
    raise
