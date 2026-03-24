"""
Classify venue names into venue_type categories for the prediction engine.

Categories: arena, outdoor, theater, sphere, festival
"""

# Exact venue name → venue_type
VENUE_TYPE_MAP: dict[str, str] = {
    # Sphere
    "Sphere": "sphere",
    # Arenas
    "Madison Square Garden": "arena",
    "Hampton Coliseum": "arena",
    "Bridgestone Arena": "arena",
    "Chase Center": "arena",
    "Climate Pledge Arena": "arena",
    "Golden 1 Center": "arena",
    "MGM Grand Garden Arena": "arena",
    "Mohegan Sun Arena": "arena",
    "Moda Center": "arena",
    "Moody Center": "arena",
    "MVP Arena": "arena",
    "Nassau Veterans Memorial Coliseum": "arena",
    "North Charleston Coliseum": "arena",
    "United Center": "arena",
    "Van Andel Arena": "arena",
    "The Forum": "arena",
    "Giant Center": "arena",
    "SNHU Arena": "arena",
    "Dunkin' Donuts Center": "arena",
    "Jerome Schottenstein Center": "arena",
    "Matthew Knight Arena": "arena",
    "Petersen Events Center": "arena",
    "Chaifetz Arena, Saint Louis University": "arena",
    "Ervin J. Nutter Center, Wright State University": "arena",
    "Credit One Stadium": "arena",
    # Outdoor / Amphitheaters
    "Dick's Sporting Goods Park": "outdoor",
    "Gorge Amphitheatre": "outdoor",
    "Alpine Valley Music Theatre": "outdoor",
    "Fenway Park": "outdoor",
    "Hersheypark Stadium": "outdoor",
    "Folsom Field": "outdoor",
    "Hollywood Bowl": "outdoor",
    "Piedmont Park": "outdoor",
    "Atlantic City Beach": "outdoor",
    "Lake Tahoe Outdoor Arena at Harveys": "outdoor",
    "Forest Hills Stadium": "outdoor",
    "Santa Barbara Bowl": "outdoor",
    "Saratoga Performing Arts Center": "outdoor",
    "Broadview Stage at SPAC": "outdoor",
    "Shoreline Amphitheatre": "outdoor",
    "Merriweather Post Pavilion": "outdoor",
    "Pine Knob Music Theatre": "outdoor",
    "Xfinity Center": "outdoor",
    "Xfinity Theatre": "outdoor",
    "Northwell Health at Jones Beach Theater": "outdoor",
    # Theaters
    "Bill Graham Civic Auditorium": "theater",
    "Met Philadelphia": "theater",
    "Bethel Woods Center for the Arts": "theater",
    "William Randolph Hearst Greek Theatre, University of California, Berkeley": "theater",
    # Festival
    "Bonnaroo Music & Arts Festival": "festival",
    "Highland Festival Grounds at Kentucky Expo Center": "festival",
    # Resort
    "Moon Palace": "outdoor",
    "Barceló Maya Beach": "outdoor",
    "BarcelÃ³ Maya Beach": "outdoor",  # encoding variant
    # TV/Special
    "NBC Television Studios, Studio 6B": "theater",
    "NPR Headquarters": "theater",
    "Rock Lititz": "theater",
}

# Keyword fallback patterns (checked in order)
VENUE_KEYWORDS: list[tuple[str, str]] = [
    ("Sphere", "sphere"),
    ("Festival", "festival"),
    ("Amphitheat", "outdoor"),  # matches Amphitheatre and Amphitheater
    ("Pavilion", "outdoor"),
    ("Park", "outdoor"),
    ("Beach", "outdoor"),
    ("Bowl", "outdoor"),
    ("Field", "outdoor"),
    ("Stadium", "outdoor"),
    ("Outdoor", "outdoor"),
    ("Coliseum", "arena"),
    ("Colosseum", "arena"),
    ("Arena", "arena"),
    ("Center", "arena"),
    ("Garden", "arena"),
    ("Forum", "arena"),
    ("Theatre", "theater"),
    ("Theater", "theater"),
    ("Auditorium", "theater"),
]


def classify_venue(venue_name: str, city: str = "") -> str:
    """Classify a venue name into a venue_type category.

    Checks exact matches first, then keyword fallback, then defaults to 'arena'.
    """
    # Exact match
    if venue_name in VENUE_TYPE_MAP:
        return VENUE_TYPE_MAP[venue_name]

    # Keyword fallback
    for keyword, vtype in VENUE_KEYWORDS:
        if keyword.lower() in venue_name.lower():
            return vtype

    return "arena"
