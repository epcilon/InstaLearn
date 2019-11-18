from InstaLearn import InstaLearn


def test_LocationandDrug():
    """
    test case for locations and drugs
    """
    il = InstaLearn()
    il.train(
        'I lived in *Munich last summer. *Germany has a relaxing, slow summer lifestyle. One night, I got food '
        'poisoning and couldn\'t find !Tylenol to make the pain go away, they insisted I take !aspirin instead.')

    _, inf_inx = il.inference(
        'When I lived in Paris last year, France was experiencing a recession. The night life was too fun, '
        'I developed an addiction to Adderall and Ritalin.')

    assert inf_inx == {'*': {5, 9}, '!': {27, 29}}
def test_DinosaurandPeriod():
    """
    test case for dinosaur and Period
    """
    il = InstaLearn()
    il.train(
        '*Sauropods first appeared in the late !Triassic Period,[7] where they somewhat resembled the closely related ('
        'and possibly ancestral) group *Prosauropoda. By the Late !Jurassic (150 million years ago), *sauropods had '
        'become widespread (especially the *diplodocids and *brachiosaurids).')

    _, inf_inx = il.inference(
        'In the Late Cretaceous, the hadrosaurs, ankylosaurs, and ceratopsians experienced success in '
        'Western North America and eastern Asia. Tyrannosaurs were present in Asia. Pachycephalosaurs were also '
        'present in both North America and Asia.')
    assert inf_inx == {'*': {7, 9, 12, 23, 29}, '!': {4}}

def test_PlanetandStar():
    """
    test case for planet and star
    """
    il = InstaLearn()
    il.train(
        'For the past month the two brightest planets, *Venus and *Jupiter, have been an eye-catching duo in the '
        'western sky after sunset. *Venus appear as a brilliant yellow planet many times brighter than any other star '
        'in the sky. It is ~18 times brighter than the brightest star !Sirius (located in the southeast) and ~75 '
        'times brighter than !Capella (the bright star located nearly over head in the evening).')

    _, inf_inx = il.inference(
        'To the right of the Moon is the Pleiades star cluster. Above and to the right is Mars. And above and to the '
        'left is the red giant star Aldebaran. By the next evening, the Moon has moved a bit higher in the sky and '
        'hangs here, above Aldebaran. The two stars that make up the front side of the pot are called "pointer stars" '
        'because they point toward the star Polaris.')
    assert inf_inx == {'*': {19}, '!': {32, 54, 80}}

if __name__ == '__main__':
    print("----------------------------------------------------------------------")
    print('test Location and Drug...')
    test_LocationandDrug()

    print("----------------------------------------------------------------------")
    print('test Dinosaur and Period...')
    test_DinosaurandPeriod()
    print("----------------------------------------------------------------------")
    print('test Planet and Star...')
    test_PlanetandStar()
