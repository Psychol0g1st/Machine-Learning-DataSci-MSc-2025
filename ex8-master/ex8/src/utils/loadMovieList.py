from os.path import join
import os
def loadMovieList():
    """
    loadMovieList reads the fixed movie list in movie.txt and returns a
    cell array of the words.
    loadMovieList reads the fixed movie list in movie.txt 
    and returns a cell array of the words in movieList.
    """
    # Read the fixed movieulary list

    with open(join(os.getcwd(), 'ex8/src/data/movie_ids.txt'),  encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movieNames = []
    for movie in movies:
        parts = movie.split()
        movieNames.append(' '.join(parts[1:]).strip())
    return movieNames
