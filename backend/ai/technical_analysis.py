"""
Technical Analysis Framework

Defines the football domain expertise for evaluating player technique.
This is the knowledge base that powers the AI Expert Coach - encoding
what constitutes good technique for each type of action.

Based on UEFA Coaching License curriculum (UEFA Pro/A/B) and the FA Four Corner Model:
- Technical: Ball manipulation, dribbling, passing, first touch, finishing
- Tactical: Decision making, awareness, spatial understanding
- Physical: Balance, coordination, speed of execution
- Psychological: Composure, confidence, concentration

References:
- UEFA Coaching Licences: https://www.uefa.com/development/coaches/uefa-coaching-licences/
- FA Four Corner Model: https://www.thefa.com/bootroom/resources/coaching/the-fas-4-corner-model
- UEFA B Diploma curriculum: https://learn.englandfootball.com/courses/football/uefa-b-licence

This is what sets us apart from VEO/HUDL - they give data, we give coaching intelligence.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SkillCategory(str, Enum):
    """Categories of football skills - aligned with UEFA B License curriculum."""
    PASSING = "passing"
    FIRST_TOUCH = "first_touch"
    SHOOTING = "shooting"
    DRIBBLING = "dribbling"
    DEFENDING = "defending"
    HEADING = "heading"
    CROSSING = "crossing"
    SET_PIECES = "set_pieces"
    POSITIONING = "positioning"
    MOVEMENT = "movement"
    DECISION_MAKING = "decision_making"
    BALL_MANIPULATION = "ball_manipulation"
    TURNING = "turning"
    ONE_V_ONE = "one_v_one"


class TechniqueAspect(str, Enum):
    """Aspects of technique to evaluate - from FA Four Corner Model Technical Corner."""
    BODY_POSITION = "body_position"
    BODY_SHAPE = "body_shape"
    HEAD_POSITION = "head_position"
    BALANCE = "balance"
    TIMING = "timing"
    CONTACT_POINT = "contact_point"
    FOLLOW_THROUGH = "follow_through"
    WEIGHT_TRANSFER = "weight_transfer"
    AWARENESS = "awareness"
    EXECUTION = "execution"
    SCANNING = "scanning"
    DECISION_SPEED = "decision_speed"


class FourCornerDomain(str, Enum):
    """The FA Four Corner Model domains for holistic player development."""
    TECHNICAL = "technical"  # Ball mastery, technique execution
    TACTICAL = "tactical"  # Decision making, game understanding
    PHYSICAL = "physical"  # Balance, coordination, speed, strength
    PSYCHOLOGICAL = "psychological"  # Confidence, composure, concentration


@dataclass
class TechniqueCheckpoint:
    """
    A specific checkpoint to evaluate in a technique.

    Based on UEFA coaching methodology - each checkpoint covers
    observable criteria that can be assessed from video analysis.
    """
    name: str
    description: str
    what_to_look_for: List[str]
    common_mistakes: List[str]
    coaching_cues: List[str]  # Short phrases coaches use (UEFA "key factors")
    importance: float = 1.0  # Weight for scoring
    four_corner_domain: FourCornerDomain = FourCornerDomain.TECHNICAL
    observable_from_video: bool = True  # Can be assessed from video


@dataclass
class SkillTechnique:
    """
    Complete technique breakdown for a skill.

    Aligned with UEFA B/A License assessment criteria and
    the FA Four Corner Model for holistic development.
    """
    skill_name: str
    category: SkillCategory
    description: str
    checkpoints: List[TechniqueCheckpoint]
    success_indicators: List[str]
    failure_indicators: List[str]
    drills_to_improve: List[str]
    pro_examples: List[str]  # "Watch how Modric does X"

    # UEFA coaching assessment criteria
    uefa_key_factors: List[str] = field(default_factory=list)  # Core coaching points
    age_appropriate_focus: Dict[str, List[str]] = field(default_factory=dict)  # Age group specific

    # Four Corner Model integration
    technical_elements: List[str] = field(default_factory=list)
    tactical_elements: List[str] = field(default_factory=list)
    physical_elements: List[str] = field(default_factory=list)
    psychological_elements: List[str] = field(default_factory=list)


# =============================================================================
# TECHNICAL KNOWLEDGE BASE
# This encodes expert football coaching knowledge
# =============================================================================

PASSING_TECHNIQUE = SkillTechnique(
    skill_name="Short Pass (Inside Foot)",
    category=SkillCategory.PASSING,
    description="The fundamental pass using the inside of the foot for accuracy over short distances",
    checkpoints=[
        TechniqueCheckpoint(
            name="Approach Angle",
            description="The angle of approach to the ball",
            what_to_look_for=[
                "Approaches ball at slight angle (not straight on)",
                "Non-kicking foot placed beside ball",
                "Hips open toward target"
            ],
            common_mistakes=[
                "Approaching too straight - limits hip rotation",
                "Standing foot too far from ball",
                "Standing foot pointing wrong direction"
            ],
            coaching_cues=["Angle your run", "Point your toe where you want it to go"],
            importance=0.8,
            four_corner_domain=FourCornerDomain.TECHNICAL
        ),
        TechniqueCheckpoint(
            name="Body Position",
            description="Overall body positioning at moment of contact",
            what_to_look_for=[
                "Slight lean over the ball",
                "Arms out for balance",
                "Head steady, eyes on ball at contact",
                "Knee of kicking leg over the ball"
            ],
            common_mistakes=[
                "Leaning back - causes ball to rise",
                "Looking up too early before contact",
                "Rigid arms affecting balance"
            ],
            coaching_cues=["Nose over toes", "Head down through contact", "Stay balanced"],
            importance=1.0,
            four_corner_domain=FourCornerDomain.PHYSICAL
        ),
        TechniqueCheckpoint(
            name="Contact Point",
            description="Where the foot strikes the ball",
            what_to_look_for=[
                "Inside of foot (arch area) contacts middle of ball",
                "Ankle locked and firm",
                "Foot at right angle to direction of pass",
                "Strikes through the center of the ball"
            ],
            common_mistakes=[
                "Toe poke instead of side foot",
                "Floppy ankle - no control",
                "Striking too high or low on ball",
                "Foot not square to target"
            ],
            coaching_cues=["Lock your ankle", "Big toe up", "Side of foot like a golf club"],
            importance=1.0,
            four_corner_domain=FourCornerDomain.TECHNICAL
        ),
        TechniqueCheckpoint(
            name="Follow Through",
            description="The continuation of the kicking motion after contact",
            what_to_look_for=[
                "Smooth follow through toward target",
                "Kicking foot continues in direction of pass",
                "Body momentum moves toward target"
            ],
            common_mistakes=[
                "Stabbing at the ball - no follow through",
                "Pulling away at contact",
                "Follow through across body"
            ],
            coaching_cues=["Stroke through the ball", "Point to your target"],
            importance=0.7,
            four_corner_domain=FourCornerDomain.TECHNICAL
        ),
        TechniqueCheckpoint(
            name="Weight of Pass",
            description="The appropriate pace on the ball for the situation",
            what_to_look_for=[
                "Pass arrives at correct speed for receiver",
                "Pace matched to distance",
                "Considers receiver's next action"
            ],
            common_mistakes=[
                "Too hard - difficult to control",
                "Too soft - intercepted or slows play",
                "Not accounting for field conditions"
            ],
            coaching_cues=["Firm but friendly", "Play it to feet", "Weight it for their run"],
            importance=0.9,
            four_corner_domain=FourCornerDomain.TACTICAL
        ),
        TechniqueCheckpoint(
            name="Pre-Pass Scanning",
            description="Awareness and scanning before receiving to pass (UEFA key factor)",
            what_to_look_for=[
                "Head movement scanning before receiving",
                "Knows where pressure is coming from",
                "Already identified passing target",
                "Body positioned to execute intended pass"
            ],
            common_mistakes=[
                "Ball watching - no awareness",
                "Deciding after receiving",
                "Not checking shoulder",
                "Unaware of pressing opponent"
            ],
            coaching_cues=["Scan before you receive", "Know your picture", "Check your shoulder"],
            importance=1.0,
            four_corner_domain=FourCornerDomain.TACTICAL
        )
    ],
    success_indicators=[
        "Ball reaches intended target",
        "Receiver can control easily with first touch",
        "Pass played with correct foot",
        "Timing allows receiver space to play",
        "Pass breaks a line of pressure",
        "Pass creates attacking opportunity"
    ],
    failure_indicators=[
        "Pass intercepted",
        "Receiver has to stretch/lunge",
        "Ball bounces awkwardly",
        "Pass behind the receiver",
        "Receiver has to check back (killing momentum)",
        "Pass too slow allowing press to engage"
    ],
    drills_to_improve=[
        "Wall passing - 100 passes each foot",
        "Triangle passing with movement",
        "One-touch passing squares",
        "Passing through gates at various distances",
        "Pressure passing - defender closing down",
        "Rondos (4v2, 5v2) for quick passing under pressure",
        "Positional play exercises with direction"
    ],
    pro_examples=[
        "Study Luka Modric's body shape when passing",
        "Watch Toni Kroos' weight of pass",
        "Observe Busquets' disguised passing",
        "Analyze Pedri's scanning before receiving"
    ],
    # UEFA coaching criteria
    uefa_key_factors=[
        "Accuracy - ball reaches intended target",
        "Weight - appropriate pace for receiver and situation",
        "Timing - released at optimal moment",
        "Disguise - minimise telegraph of pass direction",
        "Selection - choosing appropriate pass type"
    ],
    age_appropriate_focus={
        "U12": ["Basic technique", "Both feet", "Accuracy over distance"],
        "U14": ["Weight of pass", "Receiving to pass", "Breaking lines"],
        "U16": ["Disguise", "Speed of play", "Decision making under pressure"],
        "U18+": ["All elements", "Positional context", "Game management"]
    },
    # Four Corner Model elements
    technical_elements=[
        "Correct striking surface (inside foot)",
        "Ankle lock",
        "Follow through direction"
    ],
    tactical_elements=[
        "Pass selection (when/where/why)",
        "Scanning and awareness",
        "Breaking lines of pressure",
        "Creating overloads"
    ],
    physical_elements=[
        "Balance at point of contact",
        "Core stability",
        "Appropriate power generation"
    ],
    psychological_elements=[
        "Composure under pressure",
        "Confidence to play forward",
        "Willingness to take responsibility"
    ]
)


FIRST_TOUCH_TECHNIQUE = SkillTechnique(
    skill_name="First Touch Control",
    category=SkillCategory.FIRST_TOUCH,
    description="""Receiving and controlling the ball with the first touch to set up the next action.
    The first touch is the most important touch - the quality of receiving will often have a
    significant impact on the next action the player takes. (UEFA/FA Coaching Methodology)""",
    checkpoints=[
        TechniqueCheckpoint(
            name="Pre-Touch Awareness (Scanning)",
            description="Scanning and awareness before receiving - UEFA KEY FACTOR",
            what_to_look_for=[
                "Head up scanning before ball arrives",
                "Multiple scans (check shoulder) before receiving",
                "Knows where pressure is coming from",
                "Already decided what to do with ball",
                "Body position adjusted based on scan"
            ],
            common_mistakes=[
                "Ball watching - no awareness of surroundings",
                "Deciding what to do after receiving",
                "Not checking shoulder",
                "Single scan only (needs 2-3 scans)"
            ],
            coaching_cues=["Scan before you receive", "Know your picture", "Check your shoulder", "Head on a swivel"],
            importance=1.0,
            four_corner_domain=FourCornerDomain.TACTICAL
        ),
        TechniqueCheckpoint(
            name="Body Shape to Receive",
            description="How the body is positioned to receive - open body shape is crucial",
            what_to_look_for=[
                "Open body shape when possible (see the field)",
                "On half-turn to see options",
                "Side-on to receive under pressure",
                "Weight on balls of feet, ready to move",
                "Hips open to field of play"
            ],
            common_mistakes=[
                "Flat footed when receiving",
                "Back to play unnecessarily",
                "Closed body shape limiting options",
                "Square to passer, back to goal",
                "Static feet, not ready to move"
            ],
            coaching_cues=["Open up", "Show for the ball on the half turn", "Get side on", "Receive on back foot"],
            importance=1.0,
            four_corner_domain=FourCornerDomain.TECHNICAL
        ),
        TechniqueCheckpoint(
            name="Touch Direction (Positive Touch)",
            description="Where the first touch takes the ball - should set up next action",
            what_to_look_for=[
                "Touch goes away from pressure",
                "Touch sets up next action (pass/shot/dribble)",
                "Touch into space to advance play",
                "Touch keeps ball in playing distance",
                "Touch out of feet ready to play"
            ],
            common_mistakes=[
                "Touch goes toward pressure/defender",
                "Touch too heavy - loses control",
                "Touch stops ball dead (no momentum)",
                "Touch goes backwards when space is forward",
                "Ball gets stuck under feet requiring extra touch"
            ],
            coaching_cues=["Touch away from pressure", "Take it in your stride", "Play forward", "Positive first touch"],
            importance=1.0,
            four_corner_domain=FourCornerDomain.TACTICAL
        ),
        TechniqueCheckpoint(
            name="Cushion Control Technique",
            description="The softness of the touch to control the ball's pace",
            what_to_look_for=[
                "Receiving surface relaxes on contact",
                "Ball stays close and controlled",
                "Withdraws foot slightly to cushion",
                "Absorbs pace appropriately",
                "Adjusts technique to ball speed"
            ],
            common_mistakes=[
                "Rigid foot - ball bounces off",
                "Too soft - ball goes under foot",
                "Not adjusting for pace of pass",
                "Snatching at the ball"
            ],
            coaching_cues=["Soft feet", "Cushion it", "Let the ball come to you", "Relax on contact"],
            importance=0.9,
            four_corner_domain=FourCornerDomain.TECHNICAL
        ),
        TechniqueCheckpoint(
            name="Surface Selection",
            description="Choosing the right part of the body to control with",
            what_to_look_for=[
                "Uses appropriate surface for ball trajectory",
                "Inside foot for ground balls (largest surface area)",
                "Thigh/chest for aerial balls",
                "Outside foot when turning away",
                "Comfortable with both feet"
            ],
            common_mistakes=[
                "Using wrong surface for the situation",
                "Always defaulting to inside of foot",
                "Not comfortable with weaker foot",
                "Poor adjustment to ball flight"
            ],
            coaching_cues=["Pick your surface", "Both feet", "Read the flight", "Adjust to the ball"],
            importance=0.7,
            four_corner_domain=FourCornerDomain.TECHNICAL
        ),
        TechniqueCheckpoint(
            name="Protection & Shielding",
            description="Protecting the ball while receiving under pressure",
            what_to_look_for=[
                "Body between defender and ball",
                "Uses arm legally for balance/awareness",
                "Receives on back foot away from defender",
                "Strong base with low center of gravity"
            ],
            common_mistakes=[
                "Receives on front foot exposing ball",
                "Allows defender to get goal side",
                "No awareness of defender position",
                "Weak in physical duels"
            ],
            coaching_cues=["Body between ball and defender", "Receive on back foot", "Feel the defender", "Strong base"],
            importance=0.8,
            four_corner_domain=FourCornerDomain.PHYSICAL
        )
    ],
    success_indicators=[
        "Ball under control with first touch",
        "Touch sets up immediate next action (fluid, mobile motion)",
        "Maintains or increases tempo of play",
        "Keeps possession under pressure",
        "Creates separation from defender",
        "Able to play forward after receiving"
    ],
    failure_indicators=[
        "Needs multiple touches to control",
        "Touch gives ball to opponent",
        "Touch forces backwards play",
        "Loses balance after touch",
        "Touch too heavy, has to chase",
        "Ball stuck under feet, slows play"
    ],
    drills_to_improve=[
        "Wall work - receive and turn",
        "Rondo with limited touches",
        "Receive under pressure exercises",
        "Aerial control - thigh, chest, foot",
        "Touch and turn boxes",
        "Receiving on the half turn patterns",
        "Scanning exercises before receiving",
        "Positional receiving with direction"
    ],
    pro_examples=[
        "Study Bernardo Silva's touch in tight spaces",
        "Watch Benzema's first touch movement",
        "Observe Messi's close control under pressure",
        "Analyze De Bruyne's scanning before receiving",
        "Watch Thiago's body shape to receive"
    ],
    # UEFA coaching criteria
    uefa_key_factors=[
        "Scanning - awareness before receiving",
        "Body shape - open to see the field",
        "Touch direction - sets up next action",
        "Touch quality - appropriate cushion",
        "Playing surface - correct selection"
    ],
    age_appropriate_focus={
        "U12": ["Basic control", "Both feet", "Ball mastery"],
        "U14": ["Receiving on half turn", "Touch away from pressure", "Scanning basics"],
        "U16": ["Receiving under pressure", "Advanced scanning", "Protection/shielding"],
        "U18+": ["All elements at speed", "Game context", "Positional receiving"]
    },
    # Four Corner Model elements
    technical_elements=[
        "Surface selection",
        "Cushion technique",
        "Ball mastery and control"
    ],
    tactical_elements=[
        "Scanning before receiving",
        "Body shape to see options",
        "Touch direction to progress play",
        "Decision making (turn/pass/hold)"
    ],
    physical_elements=[
        "Balance and stability",
        "Coordination",
        "Strength to hold off opponent",
        "Agility to adjust position"
    ],
    psychological_elements=[
        "Composure under pressure",
        "Confidence to receive in tight areas",
        "Concentration on ball flight",
        "Bravery to show for the ball"
    ]
)


SHOOTING_TECHNIQUE = SkillTechnique(
    skill_name="Shooting (Instep Drive)",
    category=SkillCategory.SHOOTING,
    description="Striking the ball with power and accuracy toward goal using the instep (laces)",
    checkpoints=[
        TechniqueCheckpoint(
            name="Approach",
            description="The run up to the ball before striking",
            what_to_look_for=[
                "Angled approach (not straight)",
                "Controlled approach speed",
                "Eyes on ball, peripheral on goal",
                "Last step is longer for power"
            ],
            common_mistakes=[
                "Running too fast into the shot",
                "Straight approach limits technique",
                "Looking at goal instead of ball",
                "Rushed, off-balance approach"
            ],
            coaching_cues=["Angle your run", "Composed approach", "Set yourself"],
            importance=0.8
        ),
        TechniqueCheckpoint(
            name="Planting Foot",
            description="Position of the non-kicking foot",
            what_to_look_for=[
                "Planted beside the ball",
                "Pointing toward target",
                "Appropriate distance from ball",
                "Firm plant for stability"
            ],
            common_mistakes=[
                "Planting too far from ball",
                "Planting behind the ball",
                "Toe pointing wrong direction",
                "Unstable plant"
            ],
            coaching_cues=["Toe to target", "Plant beside it", "Strong base"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Striking Technique",
            description="The mechanics of the actual strike",
            what_to_look_for=[
                "Locked ankle, toe pointed down",
                "Knee over the ball at contact",
                "Strike through center of ball",
                "Laces (instep) contact",
                "Hip drives through"
            ],
            common_mistakes=[
                "Toe poke",
                "Leaning back - ball goes high",
                "Ankle not locked - no power",
                "Off-center contact - slice/hook"
            ],
            coaching_cues=["Laces through it", "Knee over ball", "Lock that ankle", "Drive through"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Body Position",
            description="Overall body shape through the strike",
            what_to_look_for=[
                "Slight forward lean",
                "Arms out for balance",
                "Head steady, down through contact",
                "Chest over the ball"
            ],
            common_mistakes=[
                "Leaning back (skied shots)",
                "Head up too early",
                "Falling away from shot",
                "Poor balance throughout"
            ],
            coaching_cues=["Stay over it", "Head down", "Attack the ball"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Follow Through",
            description="The completion of the striking motion",
            what_to_look_for=[
                "Full follow through toward target",
                "Lands on shooting foot",
                "Body momentum through the ball",
                "High finish for driven shots"
            ],
            common_mistakes=[
                "Stopping at contact",
                "Pulling out of the shot",
                "Falling away",
                "Abbreviated follow through"
            ],
            coaching_cues=["Finish high", "Through the ball", "Land on your laces"],
            importance=0.8
        ),
        TechniqueCheckpoint(
            name="Shot Selection",
            description="Choosing the right type of shot for the situation",
            what_to_look_for=[
                "Power vs placement decision appropriate",
                "Reads goalkeeper position",
                "Picks correct corner",
                "Adjusts technique to ball state"
            ],
            common_mistakes=[
                "Always going for power",
                "Not reading the goalkeeper",
                "Wrong shot type for situation",
                "Predictable shot selection"
            ],
            coaching_cues=["Pick your spot", "Read the keeper", "Placement beats power"],
            importance=0.9
        )
    ],
    success_indicators=[
        "Shot on target",
        "Good combination of power and accuracy",
        "Goalkeeper troubled/beaten",
        "Clean strike",
        "Appropriate shot selection"
    ],
    failure_indicators=[
        "Shot off target (wide or over)",
        "Weak shot, easy save",
        "Mishit/scuffed",
        "Wrong decision (pass was better)",
        "Telegraphed intention"
    ],
    drills_to_improve=[
        "Shooting from edge of box",
        "One-touch finishing",
        "Shooting after combination play",
        "Finishing under pressure/fatigue",
        "Target practice - corners",
        "Volleys and half-volleys"
    ],
    pro_examples=[
        "Study Ronaldo's body shape when shooting",
        "Watch Haaland's movement before shooting",
        "Observe Salah's placement finishing"
    ]
)


DEFENDING_1V1_TECHNIQUE = SkillTechnique(
    skill_name="1v1 Defending",
    category=SkillCategory.DEFENDING,
    description="Individual defending technique when isolated against an attacker with the ball",
    checkpoints=[
        TechniqueCheckpoint(
            name="Closing Down",
            description="The approach to the attacker",
            what_to_look_for=[
                "Quick approach while ball travels",
                "Slows down as gets closer",
                "Arrives as attacker takes touch",
                "Doesn't dive in"
            ],
            common_mistakes=[
                "Arrives too fast, can't stop",
                "Arrives too slow, attacker has time",
                "Diving in recklessly",
                "Standing off too much"
            ],
            coaching_cues=["Fast feet, slow down", "Arrive with the touch", "Don't dive in"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Body Position",
            description="Defensive stance when engaged",
            what_to_look_for=[
                "Low center of gravity",
                "Side-on stance",
                "On balls of feet, ready to react",
                "Arms out for balance (not fouling)",
                "Jockeying distance appropriate"
            ],
            common_mistakes=[
                "Too upright - easily beaten",
                "Square to attacker - can go either way",
                "Flat footed",
                "Too tight - turned easily",
                "Too far - attacker has time"
            ],
            coaching_cues=["Get low", "Side on", "Stay on your feet", "Show them one way"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Showing Direction",
            description="Forcing the attacker where you want them to go",
            what_to_look_for=[
                "Shows attacker to weaker foot",
                "Shows attacker away from goal",
                "Body angle forces direction",
                "Denies preferred route"
            ],
            common_mistakes=[
                "Letting attacker choose direction",
                "Showing toward goal",
                "Showing onto stronger foot",
                "Not committing to a direction"
            ],
            coaching_cues=["Show them outside", "Take away their strong side", "Make them predictable"],
            importance=0.9
        ),
        TechniqueCheckpoint(
            name="Patience",
            description="Waiting for the right moment to engage",
            what_to_look_for=[
                "Waits for heavy touch",
                "Doesn't commit until certain",
                "Delays attacker's progress",
                "Waits for help to arrive"
            ],
            common_mistakes=[
                "Diving in early",
                "Lunging at the ball",
                "Committing when uncertain",
                "Impatient, leaves feet"
            ],
            coaching_cues=["Stay patient", "Don't buy the feint", "Wait for your moment"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Winning the Ball",
            description="The actual tackle or interception",
            what_to_look_for=[
                "Times tackle on attacker's heavy touch",
                "Goes through the ball",
                "Stays on feet when possible",
                "Wins ball cleanly",
                "Second defender ready if beaten"
            ],
            common_mistakes=[
                "Mistimes tackle",
                "Goes to ground unnecessarily",
                "Pulls out of tackle",
                "Wins ball but gives foul"
            ],
            coaching_cues=["Win it clean", "Go through the ball", "Stay on your feet"],
            importance=0.9
        )
    ],
    success_indicators=[
        "Wins the ball",
        "Forces attacker backwards or sideways",
        "Delays until help arrives",
        "Attacker loses possession",
        "No foul committed"
    ],
    failure_indicators=[
        "Beaten by attacker",
        "Commits foul",
        "Attacker gets shot/cross off",
        "Attacker plays past defender",
        "Dives in and misses"
    ],
    drills_to_improve=[
        "1v1 channel defending",
        "Shadow defending footwork",
        "Closing down exercises",
        "Defensive reaction drills",
        "2v1 recovery defending"
    ],
    pro_examples=[
        "Study Van Dijk's body positioning",
        "Watch Marquinhos' timing of tackles",
        "Observe Kante's anticipation"
    ]
)


DRIBBLING_TECHNIQUE = SkillTechnique(
    skill_name="Dribbling & Ball Carrying",
    category=SkillCategory.DRIBBLING,
    description="Moving with the ball under control while evading opponents",
    checkpoints=[
        TechniqueCheckpoint(
            name="Ball Control",
            description="Keeping the ball close while moving",
            what_to_look_for=[
                "Ball stays within playing distance",
                "Multiple small touches",
                "Uses all surfaces of foot",
                "Ball moves with body rhythm"
            ],
            common_mistakes=[
                "Touch too heavy - ball runs away",
                "Too many touches slowing down",
                "Ball gets stuck under feet",
                "Only using one foot"
            ],
            coaching_cues=["Keep it close", "Small touches", "Both feet"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Head Position & Awareness",
            description="Seeing the field while dribbling",
            what_to_look_for=[
                "Head up scanning",
                "Uses peripheral vision for ball",
                "Knows where defenders are",
                "Sees passing options"
            ],
            common_mistakes=[
                "Head down, ball watching",
                "No awareness of pressure",
                "Misses open teammates",
                "Dribbles into trouble"
            ],
            coaching_cues=["Head up", "Scan while you carry", "Know your options"],
            importance=1.0
        ),
        TechniqueCheckpoint(
            name="Change of Pace",
            description="Using speed variations to beat defenders",
            what_to_look_for=[
                "Explosive acceleration",
                "Can slow down to draw defender",
                "Timing of acceleration correct",
                "Bursts past when defender commits"
            ],
            common_mistakes=[
                "Same pace throughout",
                "Accelerates too early",
                "Can't change speed quickly",
                "Predictable speed patterns"
            ],
            coaching_cues=["Change of pace", "Slow to fast", "Explode past them"],
            importance=0.9
        ),
        TechniqueCheckpoint(
            name="Change of Direction",
            description="Using direction changes to evade",
            what_to_look_for=[
                "Sharp cuts to wrong-foot defender",
                "Low center of gravity when turning",
                "Ball protected during direction change",
                "Uses feints and body movements"
            ],
            common_mistakes=[
                "Telegraphs direction change",
                "Loses balance when cutting",
                "Ball exposed during turn",
                "Predictable patterns"
            ],
            coaching_cues=["Sell the fake", "Low and sharp", "Protect the ball"],
            importance=0.9
        ),
        TechniqueCheckpoint(
            name="Decision Making",
            description="When to dribble vs when to pass",
            what_to_look_for=[
                "Dribbles when it advances play",
                "Releases ball when teammate open",
                "Doesn't dribble into trouble",
                "Takes on defenders in right areas"
            ],
            common_mistakes=[
                "Over-dribbling when pass is on",
                "Dribbling in dangerous areas",
                "Holding ball too long",
                "Not taking players on when should"
            ],
            coaching_cues=["Play what you see", "Don't force it", "Right time, right place"],
            importance=1.0
        )
    ],
    success_indicators=[
        "Beats defender cleanly",
        "Maintains possession",
        "Creates space for self or teammates",
        "Advances play toward goal",
        "Draws fouls in good areas"
    ],
    failure_indicators=[
        "Loses possession",
        "Dribbles into dead end",
        "Better pass option ignored",
        "Tackled/dispossessed",
        "Slows down team attack"
    ],
    drills_to_improve=[
        "Cone dribbling patterns",
        "1v1 attacking",
        "Dribbling circuits with finishing",
        "Close control in tight spaces",
        "Speed dribbling races"
    ],
    pro_examples=[
        "Study Messi's close control and change of pace",
        "Watch Vinicius Jr's explosive acceleration",
        "Observe Neymar's feints and creativity"
    ]
)


# =============================================================================
# TECHNIQUE LIBRARY - All skills available for analysis
# =============================================================================

TECHNIQUE_LIBRARY: Dict[str, SkillTechnique] = {
    "short_pass": PASSING_TECHNIQUE,
    "first_touch": FIRST_TOUCH_TECHNIQUE,
    "shooting": SHOOTING_TECHNIQUE,
    "1v1_defending": DEFENDING_1V1_TECHNIQUE,
    "dribbling": DRIBBLING_TECHNIQUE,
}


# =============================================================================
# POSITION-SPECIFIC REQUIREMENTS
# =============================================================================

@dataclass
class PositionRequirements:
    """Technical requirements for a specific position."""
    position: str
    primary_skills: List[str]
    secondary_skills: List[str]
    key_attributes: List[str]
    common_weaknesses: List[str]


POSITION_REQUIREMENTS: Dict[str, PositionRequirements] = {
    "goalkeeper": PositionRequirements(
        position="Goalkeeper",
        primary_skills=["shot_stopping", "distribution", "claiming_crosses"],
        secondary_skills=["footwork", "communication", "sweeping"],
        key_attributes=["Reflexes", "Positioning", "Decision making", "Command of area"],
        common_weaknesses=["Distribution under pressure", "Dealing with back passes", "Coming for crosses"]
    ),
    "center_back": PositionRequirements(
        position="Center Back",
        primary_skills=["1v1_defending", "heading", "passing"],
        secondary_skills=["reading_game", "communication", "ball_playing"],
        key_attributes=["Positioning", "Aerial ability", "Tackling", "Composure on ball"],
        common_weaknesses=["Turning with ball", "Playing under pressure", "Recovery pace"]
    ),
    "full_back": PositionRequirements(
        position="Full Back",
        primary_skills=["1v1_defending", "crossing", "overlapping"],
        secondary_skills=["first_touch", "passing", "dribbling"],
        key_attributes=["Stamina", "Pace", "Crossing", "Defensive awareness"],
        common_weaknesses=["Defensive positioning", "End product", "Decision in final third"]
    ),
    "defensive_midfielder": PositionRequirements(
        position="Defensive Midfielder",
        primary_skills=["passing", "1v1_defending", "positioning"],
        secondary_skills=["first_touch", "interceptions", "reading_game"],
        key_attributes=["Game reading", "Passing range", "Tackling", "Positional discipline"],
        common_weaknesses=["Mobility", "Carrying ball forward", "Playing forward passes"]
    ),
    "central_midfielder": PositionRequirements(
        position="Central Midfielder",
        primary_skills=["passing", "first_touch", "dribbling"],
        secondary_skills=["shooting", "defending", "movement"],
        key_attributes=["Vision", "Technical ability", "Work rate", "Versatility"],
        common_weaknesses=["Defensive contribution", "Goals from midfield", "Physical duels"]
    ),
    "attacking_midfielder": PositionRequirements(
        position="Attacking Midfielder",
        primary_skills=["passing", "first_touch", "dribbling", "shooting"],
        secondary_skills=["movement", "creativity", "decision_making"],
        key_attributes=["Creativity", "Final ball", "Movement", "Technical ability"],
        common_weaknesses=["Defensive work rate", "Consistency", "Physical presence"]
    ),
    "winger": PositionRequirements(
        position="Winger",
        primary_skills=["dribbling", "crossing", "shooting"],
        secondary_skills=["first_touch", "movement", "1v1_defending"],
        key_attributes=["Pace", "1v1 ability", "End product", "Directness"],
        common_weaknesses=["Defensive duties", "Decision making", "Weak foot"]
    ),
    "striker": PositionRequirements(
        position="Striker",
        primary_skills=["shooting", "first_touch", "movement"],
        secondary_skills=["heading", "hold_up_play", "pressing"],
        key_attributes=["Finishing", "Movement", "Composure", "Aerial ability"],
        common_weaknesses=["Link-up play", "Work rate out of possession", "Playing with back to goal"]
    )
}


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def get_technique_for_action(action_type: str) -> Optional[SkillTechnique]:
    """Get the technique breakdown for a specific action type."""
    action_mapping = {
        "pass": "short_pass",
        "short_pass": "short_pass",
        "long_pass": "short_pass",  # Similar technique, different power
        "touch": "first_touch",
        "first_touch": "first_touch",
        "control": "first_touch",
        "shot": "shooting",
        "shooting": "shooting",
        "finish": "shooting",
        "tackle": "1v1_defending",
        "defending": "1v1_defending",
        "1v1": "1v1_defending",
        "dribble": "dribbling",
        "dribbling": "dribbling",
        "take_on": "dribbling",
    }

    skill_key = action_mapping.get(action_type.lower())
    if skill_key:
        return TECHNIQUE_LIBRARY.get(skill_key)
    return None


def get_position_requirements(position: str) -> Optional[PositionRequirements]:
    """Get technical requirements for a position."""
    position_mapping = {
        "gk": "goalkeeper",
        "goalkeeper": "goalkeeper",
        "cb": "center_back",
        "center_back": "center_back",
        "centre_back": "center_back",
        "rb": "full_back",
        "lb": "full_back",
        "full_back": "full_back",
        "fullback": "full_back",
        "cdm": "defensive_midfielder",
        "dm": "defensive_midfielder",
        "defensive_mid": "defensive_midfielder",
        "cm": "central_midfielder",
        "central_mid": "central_midfielder",
        "cam": "attacking_midfielder",
        "am": "attacking_midfielder",
        "attacking_mid": "attacking_midfielder",
        "lw": "winger",
        "rw": "winger",
        "winger": "winger",
        "wide": "winger",
        "st": "striker",
        "cf": "striker",
        "striker": "striker",
        "forward": "striker"
    }

    position_key = position_mapping.get(position.lower())
    if position_key:
        return POSITION_REQUIREMENTS.get(position_key)
    return None


def get_coaching_points_for_mistake(action_type: str, mistake_description: str) -> List[str]:
    """Get relevant coaching cues for a specific mistake."""
    technique = get_technique_for_action(action_type)
    if not technique:
        return []

    coaching_points = []
    mistake_lower = mistake_description.lower()

    for checkpoint in technique.checkpoints:
        # Check if any common mistakes match
        for common_mistake in checkpoint.common_mistakes:
            if any(word in mistake_lower for word in common_mistake.lower().split()):
                coaching_points.extend(checkpoint.coaching_cues)
                break

    return list(set(coaching_points))  # Remove duplicates


def get_drills_for_weakness(weakness_category: str) -> List[str]:
    """Get recommended drills for a weakness area."""
    drills = []

    for technique in TECHNIQUE_LIBRARY.values():
        if weakness_category.lower() in technique.category.value:
            drills.extend(technique.drills_to_improve)

    return drills
