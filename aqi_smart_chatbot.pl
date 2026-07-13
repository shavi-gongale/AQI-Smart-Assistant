

% main rule: maps query + AQI → response
chat(AQI, Query, Response) :-
    category(AQI, Cat),                % derive category
    string_lower(Query, Q),            % normalize input

    (
        intent(outdoor, Q) -> outside(Cat, Response);
        intent(mask, Q) -> mask(Cat, Response);
        intent(health, Q) -> health(Cat, Response);
        intent(exercise, Q) -> exercise(Cat, Response);
        intent(children, Q) -> children(Cat, Response);
        intent(elder, Q) -> elder(Cat, Response);
        intent(pregnant, Q) -> pregnant(Cat, Response);
        intent(indoor, Q) -> indoor(Cat, Response);
        intent(air, Q) -> air_quality(Cat, Response);
        intent(cause, Q) -> cause(Response);
        intent(improve, Q) -> improve(Response);
        intent(reduce, Q) -> reduce(Response);
        intent(precaution, Q) -> precaution(Cat, Response);
        intent(weather, Q) -> weather(Response);
        intent(mask_type, Q) -> Response = "N95 mask is best for polluted air.";
        intent(symptoms, Q) -> symptoms(Response);

        Response = "Air quality is concerning. Stay cautious."
    ).

% intent detection using substring matching
intent(outdoor, Q) :-
    sub_string(Q, _, _, _, "outside");
    sub_string(Q, _, _, _, "go out");
    sub_string(Q, _, _, _, "go outside");
    sub_string(Q, _, _, _, "step out").

intent(mask, Q) :-
    sub_string(Q, _, _, _, "mask");
    sub_string(Q, _, _, _, "wear mask");
    sub_string(Q, _, _, _, "face cover").

intent(mask_type, Q) :-
    sub_string(Q, _, _, _, "n95");
    sub_string(Q, _, _, _, "which mask").

intent(health, Q) :-
    sub_string(Q, _, _, _, "health");
    sub_string(Q, _, _, _, "effect");
    sub_string(Q, _, _, _, "impact").

intent(symptoms, Q) :-
    sub_string(Q, _, _, _, "symptom");
    sub_string(Q, _, _, _, "cough");
    sub_string(Q, _, _, _, "breathing").

intent(exercise, Q) :-
    sub_string(Q, _, _, _, "exercise");
    sub_string(Q, _, _, _, "gym");
    sub_string(Q, _, _, _, "run").

intent(children, Q) :-
    sub_string(Q, _, _, _, "child");
    sub_string(Q, _, _, _, "kids");
    sub_string(Q, _, _, _, "baby").

intent(elder, Q) :-
    sub_string(Q, _, _, _, "elder");
    sub_string(Q, _, _, _, "old").

intent(pregnant, Q) :-
    sub_string(Q, _, _, _, "pregnant").

intent(indoor, Q) :-
    sub_string(Q, _, _, _, "indoor");
    sub_string(Q, _, _, _, "home").

intent(air, Q) :-
    sub_string(Q, _, _, _, "air");
    sub_string(Q, _, _, _, "aqi").

intent(cause, Q) :-
    sub_string(Q, _, _, _, "why");
    sub_string(Q, _, _, _, "cause").

intent(improve, Q) :-
    sub_string(Q, _, _, _, "improve");
    sub_string(Q, _, _, _, "solution").

intent(reduce, Q) :-
    sub_string(Q, _, _, _, "reduce").

intent(precaution, Q) :-
    sub_string(Q, _, _, _, "precaution");
    sub_string(Q, _, _, _, "care").

intent(weather, Q) :-
    sub_string(Q, _, _, _, "weather").

% response facts (category → message)

outside(good, "Safe to go outside.").
outside(moderate, "Mostly safe outside.").
outside(unhealthy, "Avoid outdoor activity.").
outside(poor, "Stay indoors if possible.").
outside(hazardous, "Do not go outside.").

% AQI category classification based on numeric value
% rules :  
category(AQI, good) :- AQI =< 50.
category(AQI, moderate) :- AQI > 50, AQI =< 100.
category(AQI, unhealthy) :- AQI > 100, AQI =< 200.
category(AQI, poor) :- AQI > 200, AQI =< 300.
category(AQI, hazardous) :- AQI > 300.

mask(good, "Mask not needed.").
mask(moderate, "Optional mask.").
mask(unhealthy, "Wear mask recommended.").
mask(poor, "Mask strongly recommended.").
mask(hazardous, "Mask is necessary.").

health(good, "No health risk.").
health(moderate, "Minor irritation possible.").
health(unhealthy, "Breathing discomfort possible.").
health(poor, "Serious respiratory issues.").
health(hazardous, "Severe health danger.").

exercise(good, "Safe for exercise.").
exercise(moderate, "Limit heavy exercise.").
exercise(unhealthy, "Avoid outdoor exercise.").
exercise(poor, "Exercise indoors.").
exercise(hazardous, "No outdoor activity.").

children(good, "Safe for children.").
children(moderate, "Monitor children.").
children(unhealthy, "Keep children indoors.").
children(poor, "Avoid exposure.").
children(hazardous, "Strict indoor stay.").

elder(good, "Safe for elderly.").
elder(moderate, "Be cautious.").
elder(unhealthy, "Avoid exposure.").
elder(poor, "Stay indoors.").
elder(hazardous, "High risk.").

pregnant(good, "Safe conditions.").
pregnant(moderate, "Take precautions.").
pregnant(unhealthy, "Avoid exposure.").
pregnant(poor, "Stay indoors.").
pregnant(hazardous, "Unsafe conditions.").

indoor(good, "Indoor air is safe.").
indoor(moderate, "Indoor air slightly better.").
indoor(unhealthy, "Use air purifier indoors.").
indoor(poor, "Keep windows closed.").
indoor(hazardous, "Stay indoors with purifier.").

air_quality(good, "Air quality is excellent.").
air_quality(moderate, "Air quality is acceptable.").
air_quality(unhealthy, "Air quality is unhealthy.").
air_quality(poor, "Air quality is poor.").
air_quality(hazardous, "Air quality is hazardous.").

precaution(good, "No precautions needed.").
precaution(moderate, "Stay hydrated and avoid long exposure.").
precaution(unhealthy, "Wear mask and limit outdoor time.").
precaution(poor, "Stay indoors and use purifier.").
precaution(hazardous, "Avoid all exposure.").

% static facts (no conditions)

cause("AQI increases due to traffic, industries, and burning fuels.").
improve("Plant trees, reduce vehicles, and control emissions.").
reduce("Use public transport and avoid burning waste.").
weather("Weather like wind and rain affects pollution levels.").
symptoms("Symptoms include coughing, breathing issues, and irritation.").