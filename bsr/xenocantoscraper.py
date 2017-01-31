from lxml import html
import requests
import urllib
import urlparse
import os

class XenoCantoScraper:

    url_base = 'http://xeno-canto.org'
    url_rand = urlparse.urljoin(url_base, '/explore/random')

    def __init__(self):
        pass

    def retrieve_random(self, samples_dir, ratings_filter = ['A']):
        """
        Scrapes the website for random samples of ratings specified by the
        ratings_filter parameter. Stores the listings in the directory specified
        by samples_dir in their corresponding
        species subdirectory.
        """

        page = requests.get(self.url_rand)
        tree = html.fromstring(page.content)

        listings = tree.xpath('//td[@class="sonobox new-species"]')
        for listing in listings:
            rating = listing.xpath('./div[@class="rating"]//li[@class="selected"]/span/text()')
            if len(rating) == 0 or rating[0] not in ratings_filter: continue

            species = listing.xpath('.//span[@class="common-name"]/a/text()')[0]

            url_dl = listing.xpath('.//a[@download]/@href')[0]
            url_dl = urlparse.urljoin(self.url_base, url_dl)

            filename = listing.xpath('.//a[@download]/@download')[0]
            target_path = os.path.join(samples_dir, species)
            if not os.path.exists(target_path): os.mkdir(target_path)
            target_path = os.path.join(target_path, filename)
            if os.path.exists(target_path): continue

            print species, '(', rating[0], ')'
            urllib.urlretrieve(url_dl, target_path)
            print '>', target_path
            print ''
